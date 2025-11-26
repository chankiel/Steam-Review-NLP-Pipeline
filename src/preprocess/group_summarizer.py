# src/preprocess/group_summarizer.py

import re
import pandas as pd


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def group_reviews(df, cleaner=clean_text, max_reviews_per_game=5):
    df = df.copy()
    df["app_name"] = df["app_name"].fillna("Unknown")

    # IMPORTANT: LIMIT REVIEWS PER GAME
    df = df.groupby(["app_id", "app_name"]).head(max_reviews_per_game)

    grouped = (
        df.groupby(["app_id", "app_name"])["review_text"]
          .apply(lambda x: " ||| ".join(x.astype(str)))
          .reset_index()
    )

    grouped["clean_text"] = grouped["review_text"].apply(cleaner)
    return grouped


def clean_text_textrank(text: str) -> str:
    text = str(text)
    
    text = text.replace("|||", ". ")
    
    text = re.sub(r"http\S+", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
