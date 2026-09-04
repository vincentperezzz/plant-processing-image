import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer import Scanner, plant_look
from src.paths import CKPT

def main():
    scanner = Scanner(CKPT, use_dictionary=False)
    folder = ROOT / "data" / "fine-tuning"
    
    files = sorted(folder.iterdir())
    out_lines = []
    out_lines.append(f"Checkpoint: {CKPT}")
    out_lines.append(f"Evaluating {len(files)} files in {folder}:\n")
    header = f"{'Filename':<15} {'Pred Crop':<10} {'Conf':<6} {'Marg':<6} {'Health':<10} {'Eggplant %':<12} {'Sili %':<10} {'Tomato %':<10} {'Fruit Hint':<10}"
    out_lines.append(header)
    out_lines.append("-" * 95)
    
    for p in files:
        if not p.is_file():
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            out_lines.append(f"{p.name:<15} ERROR: {e}")
            continue
            
        res = scanner.scan(img)
        scores = res.get("crop_scores", {})
        eggplant_pct = f"{scores.get('eggplant', 0.0)*100:.1f}%"
        sili_pct = f"{scores.get('sili', 0.0)*100:.1f}%"
        tomato_pct = f"{scores.get('tomato', 0.0)*100:.1f}%"
        
        line = f"{p.name:<15} {res['crop']:<10} {res['crop_confidence']:<6.2f} {res['crop_margin']:<6.2f} {res['health']:<10} {eggplant_pct:<12} {sili_pct:<10} {tomato_pct:<10} {str(res.get('fruit_hint')):<10}"
        out_lines.append(line)
        
    content = "\n".join(out_lines)
    print(content)
    (ROOT / "training" / "sili_eval.txt").write_text(content, encoding="utf-8")

if __name__ == "__main__":
    main()
