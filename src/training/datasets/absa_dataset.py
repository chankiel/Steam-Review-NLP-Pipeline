import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.utils.config import get_config

config = get_config()

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

class ABSADataset(Dataset):
    def __init__(self, df):
        self.texts = df["review_text"].tolist()
        self.labels = df[config.ASPECTS].values.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.long)

        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LEN,
            return_tensors="pt"
        )

        return {
            "ids": enc["input_ids"].squeeze(0),
            "mask": enc["attention_mask"].squeeze(0),
            "labels": labels
        }