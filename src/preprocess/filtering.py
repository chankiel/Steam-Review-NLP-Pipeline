import pandas as pd
import re

def count_unique_words(text):
    # lowercase
    text = text.lower()
    # tokenize (letters and numbers)
    words = re.findall(r"\b\w+\b", text)
    return len(set(words))


def filter_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Remove null app_name or review_text
    df = df.dropna(subset=["app_name","review_text"])
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Remove reviews with less than 10 words
    df['review_word_count'] = df['review_text'].apply(lambda x: len(str(x).split()))
    df = df[df["review_word_count"] > 10]
    
    # Remove reviews with small unique ratio words (possible trolls)
    df["review_unique_word_count"] = df["review_text"].astype(str).apply(count_unique_words)
    df["unique_ratio"] = df["review_unique_word_count"] / df["review_word_count"]
    cutoff = df["unique_ratio"].quantile(0.05)
    df = df[df["unique_ratio"] >= cutoff]
    
    # Keep reviews that are helpful
    df = df[df["review_votes"] == 1]
    
    return df