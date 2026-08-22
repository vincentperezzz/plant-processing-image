import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dictionary import PlantDictionary


def main() -> None:
    d = PlantDictionary()
    d._encode_texts()
    print("cached", len(d._index), "phrases ->", "models/dictionary_clip.pt")


if __name__ == "__main__":
    main()
