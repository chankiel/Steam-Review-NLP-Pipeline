import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.training.datasets.absa_dataset import ABSADataset 
from src.training.models.absa_model import ABSAModel
from src.utils.config import get_config
from tqdm import tqdm
import torch.nn as nn
import os

config = get_config()

def multi_aspect_loss(logits, labels):
    """
    logits: (batch, aspects, classes)
    labels: (batch, aspects)
    """
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0

    for i in range(labels.size(1)):
        total_loss += loss_fn(logits[:, i, :], labels[:, i])

    return total_loss / labels.size(1)

def train(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    train_loader = DataLoader(ABSADataset(train_df), batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(ABSADataset(valid_df), batch_size=config.VALID_BATCH_SIZE)

    # Model
    model = ABSAModel()
    model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=config.LR)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        # Wrap train_loader with tqdm
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", leave=False):
            ids = batch["ids"].cuda()
            mask = batch["mask"].cuda()
            labels = batch["labels"].cuda()

            logits = model(ids, mask)
            loss = multi_aspect_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        evaluate(model, valid_loader)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/absa_model.pt")
    
    return model

def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["ids"].cuda()
            mask = batch["mask"].cuda()
            labels = batch["labels"].cuda()

            logits = model(ids, mask)
            preds = logits.argmax(dim=2)

            correct += (preds == labels).sum().item()
            total += labels.numel()

    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")
