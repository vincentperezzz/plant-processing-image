import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from finetune_other import (
    BATCH,
    CROP_NAMES,
    HEALTH_NAMES,
    MixSet,
    NEG_MANIFEST,
    class_weights,
    load_frames,
)
from src.infer import save_meta
from src.labels import load_spec
from src.model import TwoHeadNet
from src.paths import CKPT, OLD_MODELS

HARD_BOOST = 12.0
POS_PER_EPOCH = 10000
NEG_PER_EPOCH = 8000


def load_model(device) -> TwoHeadNet:
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TwoHeadNet(len(CROP_NAMES), len(HEALTH_NAMES))
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    print("loaded", CKPT, "missing", missing, "unexpected", unexpected)
    return model.to(device)


@torch.no_grad()
def mine_hard(model, neg_df: pd.DataFrame, device) -> set[str]:
    other_i = CROP_NAMES.index("other")
    loader = DataLoader(MixSet(neg_df.assign(crop="other", health=""), False), batch_size=32, shuffle=False)
    hard = set()
    idx = 0
    model.eval()
    for x, _y_c, _y_h in tqdm(loader, desc="mine hard"):
        x = x.to(device, non_blocking=True)
        c_log, _h, _g = model(x)
        p = torch.softmax(c_log, dim=1)
        pred = p.argmax(1)
        other_p = p[:, other_i]
        for j in range(x.size(0)):
            path = str(neg_df.iloc[idx]["path"])
            if int(pred[j]) != other_i or float(other_p[j]) < 0.55:
                hard.add(path)
            idx += 1
    print("hard negatives", len(hard), "/", len(neg_df))
    return hard


def sampler_for(df: pd.DataFrame) -> WeightedRandomSampler:
    key = df["crop"].astype(str) + "|" + df["health"].fillna("").astype(str)
    counts = key.value_counts()
    w = key.map(lambda k: 1.0 / counts[k]).to_numpy() * df["hard"].map(lambda h: HARD_BOOST if h else 1.0).to_numpy()
    n_pos = int((df["crop"] != "other").sum())
    n_neg = len(df) - n_pos
    num = min(POS_PER_EPOCH, n_pos) + min(NEG_PER_EPOCH, n_neg)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=num, replacement=True)


def run_epoch(model, loader, opt, scaler, crop_w, health_w, train, device):
    model.train(train)
    total, n = 0.0, 0
    crop_ys, crop_ps = [], []
    gate_ys, gate_ps = [], []
    health_ys, health_ps = [], []
    other_i = CROP_NAMES.index("other")
    crop_ce = nn.CrossEntropyLoss(weight=crop_w)
    health_ce = nn.CrossEntropyLoss(weight=health_w, ignore_index=-1)
    bce = nn.BCEWithLogitsLoss()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y_c, y_h in tqdm(loader, leave=False):
            x = x.to(device, non_blocking=True)
            y_c = y_c.to(device)
            y_h = y_h.to(device)
            y_g = (y_c != other_i).float()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                c_log, h_log, g_log = model(x)
                loss = crop_ce(c_log, y_c) + 1.4 * bce(g_log.squeeze(1), y_g)
                if (y_h >= 0).any():
                    loss = loss + health_ce(h_log, y_h)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            bs = x.size(0)
            total += float(loss.item()) * bs
            n += bs
            crop_ys.append(y_c.detach().cpu())
            crop_ps.append(c_log.argmax(1).detach().cpu())
            health_ys.append(y_h.detach().cpu())
            health_ps.append(h_log.argmax(1).detach().cpu())
            gate_ys.append(y_g.detach().cpu())
            gate_ps.append(torch.sigmoid(g_log.squeeze(1)).detach().cpu())
    cy = torch.cat(crop_ys)
    cp = torch.cat(crop_ps)
    gy = torch.cat(gate_ys)
    gp = torch.cat(gate_ps)
    hy = torch.cat(health_ys)
    hp = torch.cat(health_ps)
    pos = cy != other_i
    neg = cy == other_i
    id_acc = (cy[pos] == cp[pos]).float().mean().item() if pos.any() else 0.0
    ood_recall = (cp[neg] == other_i).float().mean().item() if neg.any() else 0.0
    false_other = (cp[pos] == other_i).float().mean().item() if pos.any() else 0.0
    gate_pos = (gp[gy > 0.5] >= 0.5).float().mean().item() if (gy > 0.5).any() else 0.0
    gate_neg = (gp[gy < 0.5] < 0.5).float().mean().item() if (gy < 0.5).any() else 0.0
    hmask = hy >= 0
    f1s = []
    for k in range(len(HEALTH_NAMES)):
        pred = (hp == k) & hmask
        gold = (hy == k) & hmask
        tp = (pred & gold).sum().item()
        fp = (pred & ~gold).sum().item()
        fn = (~pred & gold).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec + 1e-8))
    present = [((hy == k) & hmask).any().item() for k in range(len(HEALTH_NAMES))]
    used = [f for f, p in zip(f1s, present) if p]
    macro = sum(used) / max(len(used), 1)
    return {
        "loss": total / max(n, 1),
        "id_crop_acc": id_acc,
        "ood_recall": ood_recall,
        "false_other": false_other,
        "gate_pos": gate_pos,
        "gate_neg": gate_neg,
        "health_macro_f1": macro,
    }


def main() -> None:
    if not NEG_MANIFEST.exists():
        raise SystemExit("Run training/download_negatives.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    OLD_MODELS.mkdir(parents=True, exist_ok=True)
    backup = OLD_MODELS / "best_before_gate.pt"
    if not backup.exists():
        shutil.copy2(CKPT, backup)
        print("backed up", backup)
    model = load_model(device)
    train_df, val_df = load_frames()
    neg_train = train_df[train_df["crop"] == "other"].reset_index(drop=True)
    hard = mine_hard(model, neg_train, device)
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["hard"] = train_df["path"].astype(str).isin(hard)
    val_df["hard"] = False
    print("train hard rows", int(train_df["hard"].sum()))
    train_loader = DataLoader(
        MixSet(train_df, True),
        batch_size=BATCH,
        sampler=sampler_for(train_df),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(MixSet(val_df, False), batch_size=BATCH, shuffle=False, num_workers=0)
    crop_w = class_weights(train_df["crop"], CROP_NAMES, device)
    health_w = class_weights(train_df.loc[train_df["crop"] != "other", "health"], HEALTH_NAMES, device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    phases = [("gate", True, 1e-3, 2), ("finetune", False, 1.5e-4, 1)]
    for name, freeze, lr, epochs in phases:
        print(f"phase {name}")
        if freeze:
            model.freeze_backbone(True)
        else:
            model.unfreeze_last(4)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        for epoch in range(1, epochs + 1):
            tr = run_epoch(model, train_loader, opt, scaler, crop_w, health_w, True, device)
            va = run_epoch(model, val_loader, opt, scaler, crop_w, health_w, False, device)
            score = (
                va["id_crop_acc"]
                + va["ood_recall"]
                + va["gate_neg"]
                + va["gate_pos"]
                - va["false_other"]
                + va["health_macro_f1"]
            )
            print(
                f"{name} {epoch}/{epochs} loss={tr['loss']:.3f} "
                f"val_id_crop={va['id_crop_acc']:.3f} val_ood={va['ood_recall']:.3f} "
                f"val_false_other={va['false_other']:.3f} "
                f"gate_pos={va['gate_pos']:.3f} gate_neg={va['gate_neg']:.3f} "
                f"val_health_f1={va['health_macro_f1']:.3f}"
            )
            if score > best:
                best = score
                spec = load_spec()
                torch.save(
                    {
                        "model": model.state_dict(),
                        "crops": CROP_NAMES,
                        "health": HEALTH_NAMES,
                        "crop_unknown_threshold": spec["crop_unknown_threshold"],
                        "metrics": va,
                    },
                    CKPT,
                )
                save_meta({"crops": CROP_NAMES, "health": HEALTH_NAMES, "val": va})
                print("saved", CKPT)
    print("gate finetune done", CKPT)


if __name__ == "__main__":
    main()
