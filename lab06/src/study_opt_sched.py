import os
import time
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

    device = get_device(base_cfg["device"])
    epochs = 10

    experiments = [
        {"name": "Adam_StepLR",    "optimizer": "adam", "lr": 0.001, "scheduler": "step"},
        {"name": "Adam_CosineLR",  "optimizer": "adam", "lr": 0.001, "scheduler": "cosine"},
        {"name": "Adam_PlateauLR", "optimizer": "adam", "lr": 0.001, "scheduler": "plateau"},
        {"name": "SGD_CosineLR",   "optimizer": "sgd",  "lr": 0.01,  "scheduler": "cosine"},
    ]

    mlflow.set_experiment(base_cfg["experiment_name"])
    summary_rows = []
    curves = {}

    for exp in experiments:
        cfg = yaml.safe_load(yaml.dump(base_cfg))
        cfg["data"]["batch_size"] = 128
        cfg["data"]["num_workers"] = 0
        cfg["training"]["epochs"] = epochs
        cfg["training"]["optimizer"] = exp["optimizer"]
        cfg["training"]["lr"] = exp["lr"]
        cfg["training"]["scheduler"] = exp["scheduler"]
        cfg["run_name"] = f"task5_{exp['name']}"

        set_seed(cfg["seed"])
        train_loader, val_loader = get_dataloaders(cfg)
        model = SelmanCNN(num_classes=cfg["model"]["num_classes"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        val_accs = []
        val_losses = []
        best_acc = 0.0
        t_start = time.perf_counter()

        print(f"\n--- Running {exp['name']} ({epochs} epochs) ---")
        with mlflow.start_run(run_name=cfg["run_name"]):
            mlflow.log_params({
                "task": "task5_opt_sched",
                "optimizer": exp["optimizer"],
                "lr": exp["lr"],
                "scheduler": exp["scheduler"],
                "batch_size": 128,
                "epochs": epochs,
            })

            for epoch in range(1, epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)

                if scheduler:
                    if exp["scheduler"] == "plateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                val_accs.append(val_acc)
                val_losses.append(val_loss)
                best_acc = max(best_acc, val_acc)

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                }, step=epoch)

                print(f"  Epoch {epoch:2d}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        total_time = time.perf_counter() - t_start
        curves[exp["name"]] = {"val_acc": val_accs, "val_loss": val_losses}
        summary_rows.append({
            "Run": exp["name"],
            "Optimizer": exp["optimizer"].upper(),
            "Scheduler": exp["scheduler"],
            "Init LR": exp["lr"],
            "Final Val Acc (%)": round(val_accs[-1], 2),
            "Best Val Acc (%)": round(best_acc, 2),
            "Final Val Loss": round(val_losses[-1], 4),
            "Total Time (s)": round(total_time, 1),
        })

    df = pd.DataFrame(summary_rows)
    print("\n=== Task 5: Optimizer & LR Scheduler Study (10 Epochs) ===")
    print(df.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/task5_opt_sched.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs_range = list(range(1, epochs + 1))
    for name, data in curves.items():
        axes[0].plot(epochs_range, data["val_acc"], marker="o", label=name)
        axes[1].plot(epochs_range, data["val_loss"], marker="s", label=name)

    axes[0].set_title("Validation Accuracy (10 Epochs)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("Validation Loss (10 Epochs)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("outputs/task5_opt_sched.png", dpi=150)
    print("\nSaved: outputs/task5_opt_sched.csv & outputs/task5_opt_sched.png")


if __name__ == "__main__":
    main()