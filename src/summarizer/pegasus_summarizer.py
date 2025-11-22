# src/summarizer/pegasus_summarizer.py

from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from summarizer.base import BaseSummarizer


class PegasusSummarizer(BaseSummarizer):
    def __init__(
        self,
        model_name: str = "sshleifer/distill-pegasus-xsum",
        max_input_tokens: int = 256,
        max_summary_tokens: int = 32,
        num_beams: int = 2,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        """
        Pegasus-based abstractive summarizer.

        Args:
            model_name: HF model name (distilled Pegasus by default).
            max_input_tokens: max input token length (after truncation).
            max_summary_tokens: max length of generated summary.
            num_beams: beam search width (2 is faster / less hallucination than 4+).
            device: 'cuda' or 'cpu'; if None, auto-detect.
            use_fp16: if True and on CUDA, run model in half precision for speed.
        """
        self.max_input_tokens = max_input_tokens
        self.max_summary_tokens = max_summary_tokens
        self.num_beams = num_beams
        self.use_fp16 = use_fp16

        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Optional FP16 on GPU for speed / memory
        if self.device == "cuda" and self.use_fp16:
            self.model = self.model.half()

        self.model.to(self.device)
        self.model.eval()

    def summarize_batch(self, texts: List[str]) -> List[str]:
        """
        Summarize a batch of texts with Pegasus.
        """
        if not texts:
            return []

        print(f"[Pegasus] Running generation on {len(texts)} items...")

        # Tokenize & move to device
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=self.max_input_tokens,
        ).to(self.device)

        with torch.no_grad():
            # Use autocast when on CUDA + fp16 for extra speed
            if self.device == "cuda" and self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model.generate(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        max_length=self.max_summary_tokens,
                        num_beams=self.num_beams,
                    )
            else:
                outputs = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_length=self.max_summary_tokens,
                    num_beams=self.num_beams,
                )

        return [
            self.tokenizer.decode(out, skip_special_tokens=True)
            for out in outputs
        ]
