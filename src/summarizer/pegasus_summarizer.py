# src/summarizer/pegasus_summarizer.py

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.summarizer.base import BaseSummarizer


class PegasusSummarizer(BaseSummarizer):
    def __init__(
        self,
        model_name: str = "google/pegasus-xsum",
        max_input_tokens: int = 512,
        max_summary_tokens: int = 64,
        device: str | None = None,
    ):
        self.max_input_tokens = max_input_tokens
        self.max_summary_tokens = max_summary_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def summarize_batch(self, texts: List[str]) -> List[str]:
        print(f"[Pegasus] Running generation on {len(texts)} items...")
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=self.max_input_tokens,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_length=self.max_summary_tokens,
                num_beams=4,
            )

        return [
            self.tokenizer.decode(out, skip_special_tokens=True)
            for out in outputs
        ]
