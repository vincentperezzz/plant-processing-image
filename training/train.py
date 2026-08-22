import hashlib
import sys
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
from src.labels import crop_index, crops, health_index, health_levels, load_spec
from src.model import TwoHeadNet
from src.paths import CKPT, DATA, MANIFEST, MODELS


class LeafSet(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool):
        self.frame = frame.reset_index(drop=True)
        if train:
            self.tf = transforms.Compose(
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
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

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
        return x, crop_index(row["crop"]), health_index(row["health"])


def grouped_split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df["group_id"].drop_duplicates().sample(frac=1.0, random_state=seed)
    n = len(groups)
    cut = int(n * 0.85)
    train_g = set(groups.iloc[:cut])
    val_g = set(groups.iloc[cut:])
    return df[df["group_id"].isin(train_g)], df[df["group_id"].isin(val_g)]


def class_weights(series: pd.Series, names: list[str], device: torch.device) -> torch.Tensor:
    counts = series.value_counts()
    w = []
    for name in names:
        c = float(counts.get(name, 0))
        w.append(0.0 if c <= 0 else 1.0 / c)
    t = torch.tensor(w, dtype=torch.float32, device=device)
    if float(t.sum()) == 0:
        return torch.ones(len(names), device=device)
    return t * (len(names) / t.sum().clamp_min(1e-8))


def metrics(crop_y, crop_p, health_y, health_p, n_health: int) -> dict:
    crop_acc = (crop_y == crop_p).float().mean().item()
    f1s = []
    for k in range(n_health):
        pred = health_p == k
        gold = health_y == k
        tp = (pred & gold).sum().item()
        fp = (pred & ~gold).sum().item()
        fn = (~pred & gold).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec + 1e-8))
    present = [(health_y == k).any().item() for k in range(n_health)]
    used = [f for f, p in zip(f1s, present) if p]
    macro = sum(used) / max(len(used), 1)
    return {"crop_acc": crop_acc, "health_macro_f1": macro, "health_f1": f1s}


def run_epoch(model, loader, opt, scaler, crop_w, health_w, train: bool, device):
    model.train(train)
    total = 0.0
    n = 0
    crop_ys, crop_ps, health_ys, health_ps = [], [], [], []
    crop_ce = nn.CrossEntropyLoss(weight=crop_w)
    health_ce = nn.CrossEntropyLoss(weight=health_w)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y_c, y_h in tqdm(loader, leave=False):
            x = x.to(device, non_blocking=True)
            y_c = y_c.to(device)
            y_h = y_h.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                c_log, h_log, _g = model(x)
                loss = crop_ce(c_log, y_c) + health_ce(h_log, y_h)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            bs = x.size(0)
            total += float(loss.item()) * bs
            n += bs
            crop_ys.append(y_c.detach().cpu())
            health_ys.append(y_h.detach().cpu())
            crop_ps.append(c_log.argmax(1).detach().cpu())
            health_ps.append(h_log.argmax(1).detach().cpu())
    cy = torch.cat(crop_ys)
    cp = torch.cat(crop_ps)
    hy = torch.cat(health_ys)
    hp = torch.cat(health_ps)
    m = metrics(cy, cp, hy, hp, len(health_levels()))
    m["loss"] = total / max(n, 1)
    return m


def sampler_for(df: pd.DataFrame) -> WeightedRandomSampler:
    key = df["crop"] + "|" + df["health"]
    counts = key.value_counts()
    w = key.map(lambda k: 1.0 / counts[k]).to_numpy()
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(df), replacement=True)


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("Missing data/manifest.csv. Run training/remap.py first.")
    df = pd.read_csv(MANIFEST)
    train_df, val_df = grouped_split(df)
    print("train", len(train_df), "val", len(val_df))
    print(train_df.groupby(["crop", "health"]).size().unstack(fill_value=0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    batch = 16
    train_ds = LeafSet(train_df, True)
    val_ds = LeafSet(val_df, False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        sampler=sampler_for(train_df),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = TwoHeadNet(len(crops()), len(health_levels())).to(device)
    crop_w = class_weights(train_df["crop"], crops(), device)
    health_w = class_weights(train_df["health"], health_levels(), device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    MODELS.mkdir(parents=True, exist_ok=True)
    best = -1.0
    phases = [("heads", True, 1e-3, 3), ("finetune", False, 3e-4, 5)]
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
            score = va["crop_acc"] + va["health_macro_f1"]
            print(
                f"{name} {epoch}/{epochs} train_loss={tr['loss']:.3f} "
                f"val_crop={va['crop_acc']:.3f} val_health_f1={va['health_macro_f1']:.3f}"
            )
            if score > best:
                best = score
                spec = load_spec()
                torch.save(
                    {
                        "model": model.state_dict(),
                        "crops": crops(),
                        "health": health_levels(),
                        "crop_unknown_threshold": spec["crop_unknown_threshold"],
                        "metrics": va,
                    },
                    CKPT,
                )
                save_meta({"crops": crops(), "health": health_levels(), "val": va})
                print("saved", CKPT)
    print("training done", CKPT)


if __name__ == "__main__":
    main()
