# evaluate.py

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from config import Config
from dataset import ABSADataset
from model import ABSAModel

def evaluate_full():
    df = pd.read_csv(Config.VALID_PATH)
    loader = DataLoader(ABSADataset(df), batch_size=32)

    model = ABSAModel().cuda()
    model.load_state_dict(torch.load(f"{Config.OUTPUT_DIR}/absa_model.pt"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].cuda()
            mask = batch["mask"].cuda()
            labels = batch["labels"]

            logits = model(ids, mask)
            preds = logits.argmax(dim=2).cpu()

            all_preds.append(preds)
            all_labels.append(labels)

    import numpy as np
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    for i, aspect in enumerate(Config.ASPECTS):
        print(f"=== {aspect.upper()} ===")
        print(classification_report(all_labels[:, i], all_preds[:, i]))

if __name__ == "__main__":
    evaluate_full()
