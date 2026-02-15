import pandas as pd
from fredapi import Fred
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_PATH = DATA_DIR / "raw_fred.parquet"

FRED_SERIES = {
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
    "FEDFUNDS": "fed_funds",
    "DGS10": "treasury_10y",
}

def main():
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    df = pd.DataFrame()

    for series_id, name in FRED_SERIES.items():
        s = fred.get_series(series_id)
        df[name] = s

    df = df.dropna()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(RAW_PATH)

if __name__ == "__main__":
    main()

