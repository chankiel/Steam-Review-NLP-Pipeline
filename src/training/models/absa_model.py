import torch
import torch.nn as nn
from transformers import AutoModel
from src.utils.config import get_config

config = get_config()

class ABSAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = AutoModel.from_pretrained(config.MODEL_NAME)

        hidden_size = self.base.config.hidden_size
        num_aspects = len(config.ASPECTS)
        num_classes = 3  # -1, 0, 1

        # one classification head for each aspect
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_classes)
            for _ in range(num_aspects)
        ])

    def forward(self, ids, mask):
        outputs = self.base(input_ids=ids, attention_mask=mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token

        logits = []
        for clf in self.classifiers:
            logits.append(clf(pooled))

        logits = torch.stack(logits, dim=1)  # (batch, aspects, classes)
        return logits
