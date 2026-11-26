import argparse
import os
import random
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

from model import SelmanCNN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_cfg)


def get_dataloaders(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_ds = datasets.FashionMNIST(
        root=cfg["data"]["data_dir"], train=True, download=True, transform=transform
    )
    val_ds = datasets.FashionMNIST(
        root=cfg["data"]["data_dir"], train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    return train_loader, val_loader


def build_optimizer(model, cfg):
    opt_name = cfg["training"]["optimizer"].lower()
    lr = cfg["training"]["lr"]
    wd = cfg["training"]["weight_decay"]
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=cfg["training"]["momentum"],
            weight_decay=wd,
        )
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def build_scheduler(optimizer, cfg):
    sched_name = cfg["training"]["scheduler"].lower()
    if sched_name == "none":
        return None
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["training"]["step_size"],
            gamma=cfg["training"]["gamma"],
        )
    elif sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )
    elif sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
    raise ValueError(f"Unsupported scheduler: {sched_name}")


def train_one_epoch(model, loader, criterion, optimizer, device, accum_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels) / accum_steps
        loss.backward()

        if step % accum_steps == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(cfg)
    model = SelmanCNN(num_classes=cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch = 1
    best_acc = 0.0

    resume_path = cfg["checkpoint"]["resume_path"]
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    os.makedirs(cfg["checkpoint"]["save_dir"], exist_ok=True)
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run(run_name=cfg["run_name"]):
        mlflow.log_params({
            "seed": cfg["seed"],
            "device": str(device),
            "batch_size": cfg["data"]["batch_size"],
            "num_workers": cfg["data"]["num_workers"],
            "epochs": cfg["training"]["epochs"],
            "optimizer": cfg["training"]["optimizer"],
            "lr": cfg["training"]["lr"],
            "scheduler": cfg["training"]["scheduler"],
            "grad_accum_steps": cfg["training"]["grad_accum_steps"],
        })
        mlflow.log_artifact(args.config)

        for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                accum_steps=cfg["training"]["grad_accum_steps"]
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            epoch_time = time.perf_counter() - t0
            throughput = len(train_loader.dataset) / epoch_time

            if scheduler:
                if cfg["training"]["scheduler"].lower() == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_s": epoch_time,
                "throughput_samples_per_s": throughput,
                "lr": current_lr,
            }, step=epoch)

            print(
                f"Epoch [{epoch}/{cfg['training']['epochs']}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.2f}s | {throughput:.0f} samples/s"
            )

            if val_acc > best_acc:
                best_acc = val_acc

            if epoch % cfg["checkpoint"]["save_every"] == 0:
                ckpt_path = os.path.join(
                    cfg["checkpoint"]["save_dir"], f"{cfg['run_name']}_epoch_{epoch}.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "best_acc": best_acc,
                }, ckpt_path)

        final_ckpt = os.path.join(
            cfg["checkpoint"]["save_dir"], f"{cfg['run_name']}_epoch_{cfg['training']['epochs']}.pt"
        )
        if os.path.exists(final_ckpt):
            mlflow.log_artifact(final_ckpt)

        mlflow.pytorch.log_model(
            model.cpu(),
            name="model",
            input_example=np.zeros((1, 1, 28, 28), dtype=np.float32),
        )


if __name__ == "__main__":
    main()
