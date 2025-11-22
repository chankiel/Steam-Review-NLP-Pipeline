# src/preprocess/sample_reviews.py

import pandas as pd
import os

RAW_PATH = "data/raw/dataset.csv"
OUTPUT_PATH = "data/processed/summarizer.csv"
REVIEWS_PER_GAME = 50


def sample_reviews_per_game(
    raw_csv: str = RAW_PATH,
    output_csv: str = OUTPUT_PATH,
    max_reviews: int = REVIEWS_PER_GAME,
):
    print(f"Loading raw dataset from: {raw_csv}")
    df = pd.read_csv(raw_csv)

    required_cols = {"app_id", "app_name", "review_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    print("Sampling up to 50 reviews per game...")

    # group by game, then sample
    sampled = (
        df.groupby(["app_id", "app_name"])
        .apply(lambda x: x.sample(n=min(len(x), max_reviews), random_state=42))
        .reset_index(drop=True)
    )

    print(f"Original rows: {len(df)}")
    print(f"Sampled rows:  {len(sampled)}")
    print(f"Games found:   {sampled[['app_id','app_name']].drop_duplicates().shape[0]}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    sampled.to_csv(output_csv, index=False)

    print(f"Saved sampled dataset to: {output_csv}")


if __name__ == "__main__":
    sample_reviews_per_game()
