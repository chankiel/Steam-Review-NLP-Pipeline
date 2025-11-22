# src/summarizer/lstm_summarizer.py

from typing import List
from src.summarizer.base import BaseSummarizer


class LSTMSummarizer(BaseSummarizer):
    def __init__(self, model_path: str = "models/lstm_summarizer.pt"):
        # TODO: load your LSTM seq2seq model & tokenizer here
        self.model_path = model_path

    def summarize_batch(self, texts: List[str]) -> List[str]:
        # TODO: implement batch inference using your LSTM model
        # For now, just return the first 200 chars as a stub
        return [t[:200] for t in texts]
