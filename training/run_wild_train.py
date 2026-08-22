import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import download_wild
import remap
import finetune_other

if __name__ == "__main__":
    download_wild.main()
    remap.main()
    finetune_other.main()
