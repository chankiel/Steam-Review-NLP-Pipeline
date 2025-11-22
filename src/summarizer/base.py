# src/summarizer/base.py

from abc import ABC, abstractmethod
from typing import List


class BaseSummarizer(ABC):
    @abstractmethod
    def summarize_batch(self, texts: List[str]) -> List[str]:
        """Summarize a batch of input texts."""
        raise NotImplementedError
