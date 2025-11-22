import re

def clean_review_text(text: str) -> str:
    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove steam unicode icons
    text = re.sub(r"[■□◆◘○●★☆♦▪▫▶◀↑↓]+", " ", text)

    # normalize repeated characters
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # remove repeated words
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)

    # remove punctuation except !?
    text = re.sub(r"[^\w\s!?]", " ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # remove excessive whitespace
    text = " ".join(text.split())
    
    return text