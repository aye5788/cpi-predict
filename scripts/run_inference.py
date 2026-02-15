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
MODEL_DIR = RUNTIME_DIR / "cpi_v1"

FEATURE_PATH = REPO_ROOT / "data" / "latest_features.parquet"
PRED_PATH = REPO_ROOT / "data" / "predictions.parquet"


# ----------------------------
# Model preparation
# ----------------------------
def prepare_model():
    # If _runtime exists as a FILE, delete it
    if RUNTIME_DIR.exists() and not RUNTIME_DIR.is_dir():
        RUNTIME_DIR.unlink()

    # Ensure runtime directory exists
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    # Remove previously extracted model directory
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    # Unzip model into runtime directory
    with zipfile.ZipFile(MODEL_ZIP, "r") as z:
        z.extractall(RUNTIME_DIR)

    return MODEL_DIR


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
