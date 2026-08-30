import json
import os
from pathlib import Path

import torch
import yaml
from torch import nn

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if log_interval:
            running_loss += loss.item() * batch_size
            running_total += targets.size(0)
            running_correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % log_interval == 0:
                step_log = {
                    "event": "step_progress",
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "total_batches": len(loader),
                    "step_loss": round(running_loss / running_total, 4),
                    "step_accuracy": round(running_correct / running_total, 4),
                }
                print(json.dumps(step_log), flush=True)
                running_loss = 0.0
                running_correct = 0
                running_total = 0

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def get_config_path() -> Path:
    config_env = os.environ.get("CONFIG_PATH")
    if config_env and Path(config_env).exists():
        return Path(config_env)

    container_path = Path("/app/configs/training_config.yaml")
    if container_path.exists():
        return container_path

    local_path = Path("configs/training_config.yaml")
    if local_path.exists():
        return local_path

    raise FileNotFoundError(
        "Training configuration file not found in environment, /app/configs, or configs/"
    )


def main():
    config_path = get_config_path()
    config = load_config(str(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "event": "training_start",
                "device": str(device),
                "config_path": str(config_path),
            }
        ),
        flush=True,
    )

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    train_loader, val_loader = get_dataloaders(
        data_dir=data_cfg.get("data_dir", "./data"),
        batch_size=training_cfg.get("batch_size", 64),
        max_train_samples=training_cfg.get("max_train_samples"),
        max_val_samples=training_cfg.get("max_val_samples"),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 0.001),
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = training_cfg.get("early_stopping_patience", 3)
    log_interval = training_cfg.get("log_interval")
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(training_cfg.get("epochs", 10)):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch,
            log_interval=log_interval,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        print(json.dumps(log_entry), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / config["output"]["model_name"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "architecture": config["model"]["architecture"],
                    "num_classes": config["model"]["num_classes"],
                },
                save_path,
            )
            print(
                json.dumps({"event": "checkpoint_saved", "path": str(save_path)}),
                flush=True,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    json.dumps({"event": "early_stopping", "epoch": epoch + 1}),
                    flush=True,
                )
                break

    print(
        json.dumps(
            {"event": "training_complete", "best_val_loss": round(best_val_loss, 4)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
