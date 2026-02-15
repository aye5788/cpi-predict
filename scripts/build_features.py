import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_PATH = DATA_DIR / "raw_fred.parquet"
FEATURE_PATH = DATA_DIR / "latest_features.parquet"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    features["cpi_yoy"] = df["cpi"].pct_change(12) * 100
    features["unemployment"] = df["unemployment"]
    features["fed_funds"] = df["fed_funds"]
    features["treasury_10y"] = df["treasury_10y"]

    features["cpi_yoy_lag1"] = features["cpi_yoy"].shift(1)
    features["cpi_yoy_lag3"] = features["cpi_yoy"].shift(3)

    return features.dropna()

def main():
    df = pd.read_parquet(RAW_PATH)
    features = build_features(df)
    features.to_parquet(FEATURE_PATH)

if __name__ == "__main__":
    main()

