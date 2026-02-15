import zipfile
import shutil
from pathlib import Path

import pandas as pd
import h2o


# ----------------------------
# Paths
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ZIP = REPO_ROOT / "model" / "cpi_v1.zip"
RUNTIME_DIR = REPO_ROOT / "model" / "_runtime"

FEATURE_PATH = REPO_ROOT / "data" / "latest_features.parquet"
PRED_PATH = REPO_ROOT / "data" / "predictions.parquet"


# ----------------------------
# Model preparation
# ----------------------------
def prepare_model():
    # If _runtime exists as a file, delete it
    if RUNTIME_DIR.exists() and not RUNTIME_DIR.is_dir():
        RUNTIME_DIR.unlink()

    # Clean runtime directory completely
    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    # Unzip model
    with zipfile.ZipFile(MODEL_ZIP, "r") as z:
        z.extractall(RUNTIME_DIR)

    # Find the actual H2O model directory (contains _model.ini)
    model_dirs = list(RUNTIME_DIR.rglob("_model.ini"))

    if not model_dirs:
        raise RuntimeError("No H2O model found after unzip")

    # _model.ini lives inside the model directory
    model_dir = model_dirs[0].parent

    return model_dir


# ----------------------------
# Main
# ----------------------------
def main():
    h2o.init(max_mem_size="2G", nthreads=2)

    model_path = prepare_model()
    model = h2o.load_model(str(model_path))

    df = pd.read_parquet(FEATURE_PATH)
    hf = h2o.H2OFrame(df)

    preds = model.predict(hf).as_data_frame()

    out = df.copy()
    out["predicted_cpi_yoy"] = preds.iloc[:, 0]

    out.to_parquet(PRED_PATH)

    h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()
