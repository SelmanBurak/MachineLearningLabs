import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import mlflow

from model import SelmanCNN
from train import (
    set_seed, get_device, get_dataloaders,
    build_optimizer, build_scheduler, train_one_epoch, evaluate
)


def main():
    with open("configs/base.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    cfg = yaml.safe_load(yaml.dump(base_cfg))
    cfg["data"]["batch_size"] = 128
    cfg["data"]["num_workers"] = 0
    cfg["training"]["epochs"] = 6
    cfg["training"]["scheduler"] = "cosine"

    device = get_device(cfg["device"])
    mlflow.set_experiment(cfg["experiment_name"])
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1) Uninterrupted baseline (Epochs 1..6)
    print("\n--- Run 1: Uninterrupted Training (Epochs 1-6) ---")
    set_seed(cfg["seed"])
    train_loader, val_loader = get_dataloaders(cfg)
    model_full = SelmanCNN(num_classes=cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    opt_full = build_optimizer(model_full, cfg)
    sched_full = build_scheduler(opt_full, cfg)

    full_hist = {"train_loss": [], "val_acc": []}
    with mlflow.start_run(run_name="task7_uninterrupted_6ep"):
        mlflow.log_params({"task": "task7_resume", "mode": "uninterrupted", "epochs": 6})
        for epoch in range(1, 7):
            tr_loss, _ = train_one_epoch(model_full, train_loader, criterion, opt_full, device)
            val_loss, val_acc = evaluate(model_full, val_loader, criterion, device)
            sched_full.step()
            full_hist["train_loss"].append(tr_loss)
            full_hist["val_acc"].append(val_acc)
            mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            print(f"  [Full] Epoch {epoch}/6 | Train Loss: {tr_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # 2a) Interrupted run (Epochs 1..3 -> save checkpoint & stop)
    print("\n--- Run 2a: Interrupted Training (Epochs 1-3, then STOP) ---")
    set_seed(cfg["seed"])
    train_loader, val_loader = get_dataloaders(cfg)
    model_part = SelmanCNN(num_classes=cfg["model"]["num_classes"]).to(device)
    opt_part = build_optimizer(model_part, cfg)
    sched_part = build_scheduler(opt_part, cfg)

    resumed_hist = {"train_loss": [], "val_acc": []}
    ckpt_path = "checkpoints/task7_interrupt_epoch_3.pt"

    with mlflow.start_run(run_name="task7_interrupted_ep1_3"):
        mlflow.log_params({"task": "task7_resume", "mode": "interrupted_part1", "stop_epoch": 3})
        for epoch in range(1, 4):
            tr_loss, _ = train_one_epoch(model_part, train_loader, criterion, opt_part, device)
            val_loss, val_acc = evaluate(model_part, val_loader, criterion, device)
            sched_part.step()
            resumed_hist["train_loss"].append(tr_loss)
            resumed_hist["val_acc"].append(val_acc)
            mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            print(f"  [Part1] Epoch {epoch}/6 | Train Loss: {tr_loss:.4f} | Val Acc: {val_acc:.2f}%")

        torch.save({
            "epoch": 3,
            "model_state_dict": model_part.state_dict(),
            "optimizer_state_dict": opt_part.state_dict(),
            "scheduler_state_dict": sched_part.state_dict(),
        }, ckpt_path)
        mlflow.log_artifact(ckpt_path)
        print(f"  >>> Training interrupted! Checkpoint saved to {ckpt_path}")

    # 2b) Resume from Epoch 3 checkpoint (Epochs 4..6)
    print("\n--- Run 2b: Resumed Training from Checkpoint (Epochs 4-6) ---")
    model_res = SelmanCNN(num_classes=cfg["model"]["num_classes"]).to(device)
    opt_res = build_optimizer(model_res, cfg)
    sched_res = build_scheduler(opt_res, cfg)

    ckpt = torch.load(ckpt_path, map_location=device)
    model_res.load_state_dict(ckpt["model_state_dict"])
    opt_res.load_state_dict(ckpt["optimizer_state_dict"])
    sched_res.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    print(f"  >>> Loaded checkpoint from epoch {ckpt['epoch']}, resuming at epoch {start_epoch}")

    with mlflow.start_run(run_name="task7_resumed_ep4_6"):
        mlflow.log_params({"task": "task7_resume", "mode": "resumed_part2", "start_epoch": start_epoch})
        for epoch in range(start_epoch, 7):
            tr_loss, _ = train_one_epoch(model_res, train_loader, criterion, opt_res, device)
            val_loss, val_acc = evaluate(model_res, val_loader, criterion, device)
            sched_res.step()
            resumed_hist["train_loss"].append(tr_loss)
            resumed_hist["val_acc"].append(val_acc)
            mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            print(f"  [Resumed] Epoch {epoch}/6 | Train Loss: {tr_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save table & plot
    rows = []
    for e in range(6):
        rows.append({
            "Epoch": e + 1,
            "Uninterrupted_TrainLoss": round(full_hist["train_loss"][e], 4),
            "Resumed_TrainLoss": round(resumed_hist["train_loss"][e], 4),
            "Uninterrupted_ValAcc": round(full_hist["val_acc"][e], 2),
            "Resumed_ValAcc": round(resumed_hist["val_acc"][e], 2),
        })
    df = pd.DataFrame(rows)
    print("\n=== Task 7: Checkpoint Recovery Verification ===")
    print(df.to_string(index=False))
    df.to_csv("outputs/task7_resume.csv", index=False)

    epochs_range = list(range(1, 7))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs_range, full_hist["train_loss"], marker="o", label="Uninterrupted (1-6)")
    axes[0].plot(epochs_range, resumed_hist["train_loss"], marker="s", linestyle="--", label="Interrupted (1-3) + Resumed (4-6)")
    axes[0].axvline(x=3, color="red", linestyle=":", label="Interrupt / Resume (Ep 3)")
    axes[0].set_title("Train Loss Continuity After Resume")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs_range, full_hist["val_acc"], marker="o", label="Uninterrupted (1-6)")
    axes[1].plot(epochs_range, resumed_hist["val_acc"], marker="s", linestyle="--", label="Interrupted (1-3) + Resumed (4-6)")
    axes[1].axvline(x=3, color="red", linestyle=":", label="Interrupt / Resume (Ep 3)")
    axes[1].set_title("Validation Accuracy Continuity After Resume")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("outputs/task7_resume.png", dpi=150)
    print("\nSaved: outputs/task7_resume.csv & outputs/task7_resume.png")


if __name__ == "__main__":
    main()