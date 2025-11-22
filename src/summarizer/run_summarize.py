# src/summarizer/run_summarize.py

import argparse
import pandas as pd
import torch

from src.preprocess.group_summarizer import group_reviews
from src.utils.batching import batch_iter
from src.summarizer.pegasus_summarizer import PegasusSummarizer
from src.summarizer.textrank_summarizer import TextRankSummarizer
from src.summarizer.lstm_summarizer import LSTMSummarizer


INPUT_CSV = "data/processed/summarizer.csv"


def get_summarizer(model_name: str):
    model_name = model_name.lower()
    if model_name == "pegasus":
        return PegasusSummarizer()
    if model_name == "textrank":
        return TextRankSummarizer()
    if model_name == "lstm":
        return LSTMSummarizer()
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pegasus", "textrank", "lstm"], required=True)
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading CSV from {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"app_id", "app_name", "review_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    print("Grouping and cleaning reviews per game...")
    grouped = group_reviews(df)
    print(f"Total games: {len(grouped)}")

    # Debug: show device here
    print("\n=== DEVICE CHECK ===")
    if torch.cuda.is_available():
        print("CUDA available → using GPU")
    else:
        print("CUDA NOT available → using CPU (VERY SLOW for Pegasus)")
    print("====================\n")

    summarizer = get_summarizer(args.model)
    summaries: list[str] = []

    print(f"Running {args.model} summarizer...")

    if args.model in ["textrank", "lstm"]:
        # Cheap models, no batching needed
        for i, text in enumerate(grouped["clean_text"].tolist(), start=1):
            if i % 100 == 0:
                print(f"Processed {i}/{len(grouped)} games...")
            summaries.extend(summarizer.summarize_batch([text]))

    else:
        # PEGASUS → needs batching
        total_batches = len(grouped) // args.batch_size + 1
        print(f"Total batches: {total_batches}\n")

        for batch_idx, idx_batch in enumerate(
            batch_iter(grouped.index.tolist(), args.batch_size),
            start=1
        ):
            print(f"=== Batch {batch_idx}/{total_batches} ===")
            texts = grouped.loc[idx_batch, "clean_text"].tolist()

            print(f"Running model on batch of size {len(texts)} ...")
            batch_summaries = summarizer.summarize_batch(texts)

            summaries.extend(batch_summaries)
            print(f"✓ Finished batch {batch_idx}/{total_batches}\n")

    grouped[f"summary_{args.model}"] = summaries
    grouped.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
