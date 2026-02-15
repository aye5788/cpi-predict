import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

RAW_PATH = DATA_DIR / "raw_fred.parquet"
FEATURE_PATH = DATA_DIR / "latest_features.parquet"

def main():
    df = pd.read_parquet(RAW_PATH)

    features = pd.DataFrame(index=df.index)

    # === EXACT FEATURE LOGIC FROM TRAINING NOTEBOOK ===
    features["cpi_yoy"] = df["cpi"].pct_change(12) * 100
    features["core_cpi_yoy"] = df["core_cpi"].pct_change(12) * 100
    features["energy_yoy"] = df["energy"].pct_change(12) * 100
    features["shelter_yoy"] = df["shelter"].pct_change(12) * 100
    features["sticky_core_yoy"] = df["sticky_core"].pct_change(12) * 100
    # =================================================

    features = features.dropna()

    DATA_DIR.mkdir(exist_ok=True)
    features.to_parquet(FEATURE_PATH)

if __name__ == "__main__":
    main()
