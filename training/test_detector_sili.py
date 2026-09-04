import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detect import find_plants_exg, YOLO_CLASSES
from src.infer import Scanner
from src.paths import CKPT

def main():
    scanner = Scanner(CKPT, use_dictionary=True)
    folder = ROOT / "data" / "fine-tuning"
    out_lines = []
    
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        img = Image.open(p).convert("RGB")
        w, h = img.size
        
        # Test full image scan
        full = scanner.scan(img)
        
        # Test plant bounding box detection
        boxes = find_plants_exg(img)
        box_preds = []
        for b in boxes:
            x1, y1, x2, y2 = b["xyxy"]
            crop_img = img.crop((x1, y1, x2, y2))
            c_res = scanner.scan(crop_img)
            box_preds.append((b["xyxy"], c_res["crop"], c_res["crop_confidence"], c_res.get("guess"), c_res.get("crop_scores", {})))
            
        out_lines.append(f"Image {p.name} ({w}x{h}):")
        out_lines.append(f"  Full scan: {full['crop']} (conf={full['crop_confidence']:.2f}, guess={full['guess']}, dict={full.get('named_plant')})")
        out_lines.append(f"  Boxes ({len(boxes)}):")
        for xyxy, crop, conf, guess, sc in box_preds:
            top_sc = " ".join(f"{k}:{v*100:.1f}%" for k,v in sorted(sc.items(), key=lambda x:-x[1])[:3])
            out_lines.append(f"    box {xyxy} -> crop={crop} (conf={conf:.2f}, guess={guess}) [{top_sc}]")
        out_lines.append("")
        
    content = "\n".join(out_lines)
    print(content)
    (ROOT / "training" / "detector_sili_eval.txt").write_text(content, encoding="utf-8")

if __name__ == "__main__":
    main()
