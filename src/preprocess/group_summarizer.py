# src/preprocess/group_summarizer.py

import re
import pandas as pd


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def group_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["app_name"] = df["app_name"].fillna("Unknown")
    df["review_text"] = df["review_text"].astype(str)

    grouped = (
        df.groupby(["app_id", "app_name"])["review_text"]
          .apply(lambda x: " ||| ".join(x.dropna().astype(str)))
          .reset_index()
          .rename(columns={"review_text": "combined_reviews"})
    )

    grouped["clean_text"] = grouped["combined_reviews"].apply(clean_text)
    return grouped
