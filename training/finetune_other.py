import hashlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.infer import save_meta
from src.labels import crops, health_levels, load_spec
from src.model import TwoHeadNet
from src.paths import CKPT, DATA, FIELD_HOLDOUT_GROUPS, MANIFEST, OLD_MODELS

NEG_MANIFEST = DATA / "negatives_manifest.csv"
CROP_NAMES = crops() + ["other"]
HEALTH_NAMES = health_levels()
BATCH = 64
POS_PER_EPOCH = 18000
NEG_PER_EPOCH = 6000

TRAIN_TF = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(18),
        transforms.ColorJitter(0.25, 0.25, 0.2, 0.04),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
VAL_TF = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MixSet(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool):
        self.frame = frame.reset_index(drop=True)
        self.tf = TRAIN_TF if train else VAL_TF

    def __len__(self) -> int:
        return len(self.frame)

    def _cached(self, src: str) -> Path:
        cache = DATA / "processed" / "256"
        cache.mkdir(parents=True, exist_ok=True)
        out = cache / (hashlib.md5(src.encode("utf-8")).hexdigest() + ".jpg")
        if not out.exists():
            im = Image.open(src).convert("RGB")
            im.thumbnail((256, 256))
            canvas = Image.new("RGB", (256, 256), (0, 0, 0))
            x = (256 - im.size[0]) // 2
            y = (256 - im.size[1]) // 2
            canvas.paste(im, (x, y))
            canvas.save(out, quality=90)
        return out

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        img = Image.open(self._cached(row["path"])).convert("RGB")
        x = self.tf(img)
        crop_i = CROP_NAMES.index(row["crop"])
        health = row["health"]
        health_i = HEALTH_NAMES.index(health) if isinstance(health, str) and health in HEALTH_NAMES else -1
        return x, crop_i, health_i


def grouped_split(df: pd.DataFrame, seed: int = 42):
    groups = df["group_id"].drop_duplicates().sample(frac=1.0, random_state=seed)
    cut = int(len(groups) * 0.85)
    train_g = set(groups.iloc[:cut])
    return df[df["group_id"].isin(train_g)], df[~df["group_id"].isin(train_g)]


def load_frames():
    pos = pd.read_csv(MANIFEST)
    neg = pd.read_csv(NEG_MANIFEST)

    def gid(row):
        kind = str(row["kind"])
        folder = str(row["class_folder"])
        if kind == "indoor":
            stem = Path(row["path"]).stem
            try:
                bucket = int(stem) % 40
            except ValueError:
                bucket = abs(hash(stem)) % 40
            return f"indoor:{bucket}"
        return f"{kind}:{folder}"

    neg = neg.assign(
        crop="other",
        health="",
        source="neg",
        group_id=neg.apply(gid, axis=1),
    )[["path", "crop", "health", "source", "group_id"]]
    pos = pos[["path", "crop", "health", "source", "group_id"]]
    if FIELD_HOLDOUT_GROUPS.exists():
        locked = {g.strip() for g in FIELD_HOLDOUT_GROUPS.read_text(encoding="utf-8").splitlines() if g.strip()}
        dropped = int(pos["group_id"].astype(str).isin(locked).sum())
        pos = pos[~pos["group_id"].astype(str).isin(locked)].copy()
        print("field holdout locked", len(locked), "photos out", dropped)
    pos_tr, pos_va = grouped_split(pos)
    neg_tr, neg_va = grouped_split(neg, seed=7)
    train = pd.concat([pos_tr, neg_tr], ignore_index=True)
    val = pd.concat([pos_va, neg_va], ignore_index=True)
    return train, val


def sampler_for(df: pd.DataFrame) -> WeightedRandomSampler:
    crop = df["crop"].astype(str)
    health = df["health"].fillna("").astype(str)
    ch = crop + "|" + health
    n_ch = ch.value_counts()
    w = ch.map(lambda k: 1.0 / float(n_ch[k])).to_numpy(dtype=float, copy=True)
    if "source" in df.columns:
        source = df["source"].fillna("na").astype(str)
        palay = crop == "palay"
        palay_df = df.loc[palay]
        for hname, sub in palay_df.groupby(palay_df["health"].fillna(""), dropna=False):
            counts = sub["source"].value_counts()
            big = counts[counts >= 100]
            if len(big) < 2:
                continue
            n_big = float(len(big))
            hkey = "" if not isinstance(hname, str) else hname
            for src_name, n in big.items():
                mask = palay & (source == src_name) & (health.fillna("") == hkey)
                w[mask.to_numpy()] = (1.0 / float(n)) / n_big
    n_pos = int((df["crop"] != "other").sum())
    num = min(POS_PER_EPOCH, n_pos) + min(NEG_PER_EPOCH, len(df) - n_pos)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=num, replacement=True)


def expand_checkpoint(device) -> TwoHeadNet:
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    old_names = blob["crops"]
    model = TwoHeadNet(len(CROP_NAMES), len(HEALTH_NAMES))
    state = blob["model"]
    if len(old_names) == len(CROP_NAMES):
        model.load_state_dict(state, strict=False)
        print("checkpoint already has other class")
        return model.to(device)
    with torch.no_grad():
        new_w = torch.zeros(len(CROP_NAMES), state["crop_head.weight"].shape[1])
        new_b = torch.zeros(len(CROP_NAMES))
        new_w[: len(old_names)] = state["crop_head.weight"]
        new_b[: len(old_names)] = state["crop_head.bias"]
        nn.init.normal_(new_w[len(old_names) :], std=0.01)
        state = dict(state)
        state["crop_head.weight"] = new_w
        state["crop_head.bias"] = new_b
    model.load_state_dict(state, strict=False)
    return model.to(device)


def pick_batch(device: torch.device) -> int:
    if device.type != "cuda":
        return 16
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dummy = TwoHeadNet(len(CROP_NAMES), len(HEALTH_NAMES)).to(device)
    dummy.train()
    opt = torch.optim.AdamW(dummy.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    chosen = 16
    for b in (192, 160, 128, 96, 64, 48, 32, 16):
        try:
            torch.cuda.empty_cache()
            opt.zero_grad(set_to_none=True)
            x = torch.randn(b, 3, 224, 224, device=device)
            y = torch.zeros(b, dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda"):
                c_log, h_log, _g = dummy(x)
                loss = nn.functional.cross_entropy(c_log, y) + nn.functional.cross_entropy(h_log, y)
            scaler.scale(loss).backward()
            opt.zero_grad(set_to_none=True)
            del x, c_log, h_log, loss
            torch.cuda.empty_cache()
            chosen = b
            break
        except torch.cuda.OutOfMemoryError:
            dummy.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    del dummy, opt, scaler
    torch.cuda.empty_cache()
    print("gpu batch", chosen)
    return chosen


def class_weights(series: pd.Series, names: list[str], device) -> torch.Tensor:
    counts = series.value_counts()
    w = []
    for name in names:
        c = float(counts.get(name, 0))
        w.append(0.0 if c <= 0 else 1.0 / c)
    t = torch.tensor(w, dtype=torch.float32, device=device)
    if float(t.sum()) == 0:
        return torch.ones(len(names), device=device)
    return t * (len(names) / t.sum().clamp_min(1e-8))


def run_epoch(model, loader, opt, scaler, crop_w, health_w, train, device):
    model.train(train)
    total, n = 0.0, 0
    crop_ys, crop_ps = [], []
    health_ys, health_ps = [], []
    crop_ce = nn.CrossEntropyLoss(weight=crop_w)
    health_ce = nn.CrossEntropyLoss(weight=health_w, ignore_index=-1)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y_c, y_h in tqdm(loader, leave=False):
            x = x.to(device, non_blocking=True)
            y_c = y_c.to(device)
            y_h = y_h.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                c_log, h_log, _g = model(x)
                loss = crop_ce(c_log, y_c)
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
    cy = torch.cat(crop_ys)
    cp = torch.cat(crop_ps)
    hy = torch.cat(health_ys)
    hp = torch.cat(health_ps)
    other_i = CROP_NAMES.index("other")
    pos = cy != other_i
    neg = cy == other_i
    id_acc = (cy[pos] == cp[pos]).float().mean().item() if pos.any() else 0.0
    ood_recall = (cp[neg] == other_i).float().mean().item() if neg.any() else 0.0
    false_other = (cp[pos] == other_i).float().mean().item() if pos.any() else 0.0
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
        "health_macro_f1": macro,
        "health_f1": f1s,
    }


def main() -> None:
    if not NEG_MANIFEST.exists():
        raise SystemExit("Run training/download_negatives.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    batch = pick_batch(device)
    workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    if device.type == "cuda" and os.name == "nt":
        workers = 2
    train_df, val_df = load_frames()
    print("train", len(train_df), "val", len(val_df), "batch", batch, "workers", workers)
    print(train_df["crop"].value_counts().to_string())
    OLD_MODELS.mkdir(parents=True, exist_ok=True)
    backup = OLD_MODELS / f"best_before_junk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    if CKPT.exists():
        shutil.copy2(CKPT, backup)
        print("backed up", backup)
    model = expand_checkpoint(device)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        MixSet(train_df, True),
        batch_size=batch,
        sampler=sampler_for(train_df),
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    val_loader = DataLoader(
        MixSet(val_df, False),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    crop_w = class_weights(train_df["crop"], CROP_NAMES, device)
    health_rows = train_df.loc[train_df["health"].isin(HEALTH_NAMES), "health"]
    health_w = class_weights(health_rows, HEALTH_NAMES, device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    phases = [("heads", True, 1e-3, 2), ("last4", False, 2e-4, 3), ("full", "full", 5e-5, 3)]
    for name, freeze, lr, epochs in phases:
        print(f"phase {name}")
        if freeze is True:
            model.freeze_backbone(True)
        elif freeze == "full":
            model.freeze_backbone(False)
        else:
            model.unfreeze_last(4)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        for epoch in range(1, epochs + 1):
            tr = run_epoch(model, train_loader, opt, scaler, crop_w, health_w, True, device)
            va = run_epoch(model, val_loader, opt, scaler, crop_w, health_w, False, device)
            score = va["id_crop_acc"] + va["ood_recall"] - va["false_other"] + va["health_macro_f1"]
            print(
                f"{name} {epoch}/{epochs} loss={tr['loss']:.3f} "
                f"val_id_crop={va['id_crop_acc']:.3f} val_ood_recall={va['ood_recall']:.3f} "
                f"val_false_other={va['false_other']:.3f} val_health_f1={va['health_macro_f1']:.3f}"
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
    print("finetune done", CKPT)


if __name__ == "__main__":
    main()
