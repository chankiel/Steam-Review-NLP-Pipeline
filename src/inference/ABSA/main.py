import torch
import os
from transformers import AutoTokenizer
from src.utils.config import get_config
from src.training.models.absa_model import ABSAModel

config = get_config()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..","..", "models", "best", "absa_model.pt")
MODEL_PATH = os.path.normpath(MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print("Loading ABSA model...")
    model = ABSAModel()
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("Model loaded.")
    return model


def analyze_review(model, text: str):
    """Run ABSA on a single string input."""

    # Tokenize into a single batch of size 1
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)       # (1, seq_len)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)     # (1, num_aspects, 3)

    preds = logits.argmax(dim=2).squeeze(0).cpu().numpy()  # (num_aspects,)
    preds = preds - 1  # convert {0,1,2} → {-1,0,1}

    return preds


def print_result(preds):
    """Pretty-print the ABSA prediction."""
    print("\n=== ABSA Result ===")

    sentiment_map = {
        -1: "negative",
         0: "neutral",
         1: "positive"
    }

    for aspect, value in zip(config.ASPECTS, preds):
        print(f"{aspect:15} → {sentiment_map[value]}")

    print("===================")


def repl(model):
    print("\nABSA CLI ready.")
    print("Type a review. Type 'quit' to exit.\n")

    while True:
        text = input("Review> ").strip()

        if text.lower() in ["quit", "exit"]:
            print("Bye!")
            break

        preds = analyze_review(model, text)
        print_result(preds)
        print()


def main():
    model = load_model()
    repl(model)


if __name__ == "__main__":
    main()
