import zipfile
import shutil
from pathlib import Path
import h2o

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ZIP = REPO_ROOT / "model" / "cpi_v1.zip"
RUNTIME_DIR = REPO_ROOT / "model" / "_runtime"
MODEL_DIR = RUNTIME_DIR / "cpi_v1"

def prepare_model():
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(MODEL_ZIP, "r") as z:
        z.extractall(RUNTIME_DIR)

    return MODEL_DIR

def main():
    h2o.init()

    model_path = prepare_model()
    model = h2o.load_model(str(model_path))

    # load features + run inference here

