import hashlib
import sys
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import DATA, MANIFEST

def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST}")
        
    df = pd.read_csv(MANIFEST)
    print(f"Original manifest rows: {len(df)}")
    
    ft_dir = DATA / "fine-tuning"
    existing_paths = set(df["path"].str.replace("\\", "/"))
    
    # Map each image in fine-tuning
    new_rows = []
    # All 13 images are sili plants from real field/garden
    # Some have healthy foliage and ripe/green chilies, some have mild spots
    for p in sorted(ft_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
            
        posix_p = str(p.resolve()).replace("\\", "/")
        if posix_p in existing_paths or str(p.resolve()) in set(df["path"]):
            print(f"Skipping already present {p.name}")
            continue
            
        stem = p.stem.lower()
        # Classify health status based on foliar condition
        # sili-3, sili-11, sili-12 show critical/mild foliar curl or spots; most others are healthy/mild
        if any(k in stem for k in ["sili-3", "sili-11", "sili-12"]):
            health = "mild"
        elif any(k in stem for k in ["sili-2", "sili-1"]):
            health = "healthy"
        else:
            health = "healthy"
            
        group_id = f"field_sili:{p.stem}"
        new_rows.append({
            "path": str(p.resolve()),
            "crop": "sili",
            "health": health,
            "source": "field_sili",
            "group_id": group_id,
            "class_folder": "fine-tuning"
        })
        
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Duplicate/oversample the new field data slightly in the manifest to give it strong presence
        # but the WeightedRandomSampler in finetune_other.py will also balance it.
        combined = pd.concat([df, new_df], ignore_index=True)
        # Backup original manifest
        backup_path = DATA / "manifest_backup.csv"
        if not backup_path.exists():
            df.to_csv(backup_path, index=False)
            print(f"Backed up original manifest to {backup_path}")
            
        combined.to_csv(MANIFEST, index=False)
        print(f"Added {len(new_rows)} new images to {MANIFEST}. New total rows: {len(combined)}")
    else:
        print("No new images to add.")
        
    # Pre-cache 256x256 thumbnails for speed during training
    cache_dir = DATA / "processed" / "256"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p in ft_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            src_str = str(p.resolve())
            out_file = cache_dir / (hashlib.md5(src_str.encode("utf-8")).hexdigest() + ".jpg")
            im = Image.open(p).convert("RGB")
            im.thumbnail((256, 256))
            canvas = Image.new("RGB", (256, 256), (0, 0, 0))
            x = (256 - im.size[0]) // 2
            y = (256 - im.size[1]) // 2
            canvas.paste(im, (x, y))
            canvas.save(out_file, quality=90)
    print("Pre-cached thumbnails in data/processed/256/")

if __name__ == "__main__":
    main()
