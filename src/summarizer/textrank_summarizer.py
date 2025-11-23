# src/summarizer/textrank_summarizer.py

from typing import List

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as _TextRank

from summarizer.base import BaseSummarizer


class TextRankSummarizer(BaseSummarizer):
    def __init__(self, sentence_count: int = 3, language: str = "english"):
        self.sentence_count = sentence_count
        self.language = language
        self.summarizer = _TextRank()

    def _summarize_one(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        sentences = self.summarizer(parser.document, self.sentence_count)
        return " ".join(str(s) for s in sentences)

    def summarize_batch(self, texts: List[str]) -> List[str]:
        return [self._summarize_one(t) for t in texts]
