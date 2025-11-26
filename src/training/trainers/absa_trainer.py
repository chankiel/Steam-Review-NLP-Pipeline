import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.training.datasets.absa_dataset import ABSADataset 
from src.training.models.absa_model import ABSAModel
from src.utils.config import get_config
from tqdm import tqdm
import torch.nn as nn

config = get_config()
device = "cuda" if torch.cuda.is_available() else "cpu"

def multi_aspect_loss(logits, labels):
    """
    logits: (batch_size, num_aspects, num_classes)
    labels: (batch_size, num_aspects), values should be 0..num_classes-1
    """
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0

    for i in range(labels.size(1)):
        if labels[:, i].min() < 0 or labels[:, i].max() >= logits.size(2):
            raise ValueError(f"Invalid labels in aspect {i}")
        total_loss += loss_fn(logits[:, i, :], labels[:, i])

    return total_loss / labels.size(1)

def train(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    train_loader = DataLoader(
        ABSADataset(train_df),
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True
    )
    valid_loader = DataLoader(
        ABSADataset(valid_df),
        batch_size=config.VALID_BATCH_SIZE
    )

    model = ABSAModel().to(device)
    optimizer = AdamW(model.parameters(), lr=float(config.LR))

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", leave=False):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = (batch["labels"] + 1).to(device)

            logits = model(ids, mask)
            loss = multi_aspect_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {avg_loss:.4f}")

        # Validation
        evaluate(model, valid_loader)

    return model


def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = (batch["labels"] + 1).to(device)
            logits = model(ids, mask)
            preds = logits.argmax(dim=2)

            correct += (preds == labels).sum().item()
            total += labels.numel()

    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")
    return acc
