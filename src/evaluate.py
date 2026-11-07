"""Training entry point for temporal grounding."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import DialogueGroundingDataset
from src.model import TemporalGroundingConfig, TemporalGroundingModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            outputs = model(batch)
            loss = criterion(outputs["logits"], batch["labels"])
            loss_sum += loss.item() * batch["labels"].size(0)
            correct += (outputs["logits"].argmax(-1) == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1)}


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config, encoding="utf-8") as handle:
        cfg = json.load(handle)
    train_set = DialogueGroundingDataset(cfg["train_split"])
    val_set = DialogueGroundingDataset(cfg["val_split"])
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"])
    model = TemporalGroundingModel(TemporalGroundingConfig(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            outputs = model(batch)
            loss = torch.nn.functional.cross_entropy(outputs["logits"], batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch} val_loss={metrics['loss']:.4f} val_acc={metrics['accuracy']:.4f}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(args.output_dir) / "temporal_grounding.pt")


if __name__ == "__main__":
    main()
