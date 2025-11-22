import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SegLM
from preprocess.group_summarizer import group_reviews

def summarize_text(model, tok, text):
    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=512
    )

    out = model.generate(
        **inputs,
        max_length=60,
        num_beams=4
    )
    return tok.decode(out[0], skip_special_tokens=True)