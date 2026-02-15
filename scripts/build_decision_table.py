import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "predictions.parquet"
OUTPUT_DIR = REPO_ROOT / "outputs"

def main():
    df = pd.read_parquet(DATA_PATH)
    latest = df.iloc[-1]

    market_price = 0.55  # placeholder – replace with real market-implied prob
    model_prob = latest["predicted_cpi_yoy"]

    edge = model_prob - market_price
    decision = "BUY" if edge > 0.05 else "PASS"

    table = pd.DataFrame([{
        "model_prob": model_prob,
        "market_prob": market_price,
        "edge": edge,
        "decision": decision
    }])

    OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_DIR / "decision_table_latest.csv", index=False)
    table.to_markdown(OUTPUT_DIR / "decision_table_latest.md", index=False)

if __name__ == "__main__":
    main()

