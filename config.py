import os
from pathlib import Path
import sys

BASE_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    print(f"Base directory is set to: {BASE_DIR}")