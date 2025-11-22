import pandas as pd

def sampling_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_rows = 10

    df = (
        df
        .groupby(["app_name", "review_score"])
        .apply(lambda g: g.sample(n=min(num_rows, len(g)), random_state=42))
        .reset_index(drop=True)
    )

    return df
