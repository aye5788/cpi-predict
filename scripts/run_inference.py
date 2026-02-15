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
    # Reset runtime directory completely
    if RUNTIME_DIR.exists():
        if RUNTIME_DIR.is_dir():
            shutil.rmtree(RUNTIME_DIR)
        else:
            RUNTIME_DIR.unlink()

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    # Unzip model
    with zipfile.ZipFile(MODEL_ZIP, "r") as z:
        z.extractall(RUNTIME_DIR)

    # Find first extracted path (file or directory)
    extracted = list(RUNTIME_DIR.iterdir())

    if not extracted:
        raise RuntimeError("Model ZIP extracted nothing")

    # If only one item, use it directly
    model_path = extracted[0]

    return model_path


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

