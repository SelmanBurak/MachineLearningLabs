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
    build_optimizer, train_one_epoch, evaluate
)


def run_config(base_cfg, device, run_name, batch_size, accum_steps, epochs=5):
    cfg = yaml.safe_load(yaml.dump(base_cfg))
    cfg["data"]["batch_size"] = batch_size
    cfg["data"]["num_workers"] = 0
    cfg["training"]["epochs"] = epochs
    cfg["training"]["grad_accum_steps"] = accum_steps
    cfg["run_name"] = run_name

    set_seed(cfg["seed"])
    train_loader, val_loader = get_dataloaders(cfg)
    model = SelmanCNN(num_classes=cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.perf_counter()

    print(f"\n--- Running {run_name} (BS={batch_size}, accum={accum_steps}, eff_BS={batch_size*accum_steps}) ---")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "task": "task6_grad_accum",
            "batch_size": batch_size,
            "grad_accum_steps": accum_steps,
            "effective_batch_size": batch_size * accum_steps,
            "epochs": epochs,
        })

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, accum_steps=accum_steps
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            print(f"  Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    elapsed = time.perf_counter() - t0
    return history, elapsed


def main():
    with open("configs/base.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    device = get_device(base_cfg["device"])
    mlflow.set_experiment(base_cfg["experiment_name"])
    epochs = 5

    hist_native, time_native = run_config(
        base_cfg, device, "task6_native_bs256", batch_size=256, accum_steps=1, epochs=epochs
    )
    hist_accum, time_accum = run_config(
        base_cfg, device, "task6_accum_bs64x4", batch_size=64, accum_steps=4, epochs=epochs
    )

    rows = []
    for e in range(epochs):
        rows.append({
            "Epoch": e + 1,
            "Native_BS256_TrainLoss": round(hist_native["train_loss"][e], 4),
            "Accum_64x4_TrainLoss": round(hist_accum["train_loss"][e], 4),
            "Native_BS256_ValAcc": round(hist_native["val_acc"][e], 2),
            "Accum_64x4_ValAcc": round(hist_accum["val_acc"][e], 2),
        })

    df = pd.DataFrame(rows)
    print("\n=== Task 6: Gradient Accumulation Comparison (Effective BS = 256) ===")
    print(df.to_string(index=False))
    print(f"\nTotal Time -> Native BS=256: {time_native:.1f}s | Accum BS=64x4: {time_accum:.1f}s")

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/task6_grad_accum.csv", index=False)

    epochs_range = list(range(1, epochs + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs_range, hist_native["train_loss"], marker="o", label="Native BS=256 (accum=1)")
    axes[0].plot(epochs_range, hist_accum["train_loss"], marker="s", linestyle="--", label="Emulated BS=64 (accum=4)")
    axes[0].set_title("Training Loss: Native vs Gradient Accumulation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs_range, hist_native["val_acc"], marker="o", label="Native BS=256 (accum=1)")
    axes[1].plot(epochs_range, hist_accum["val_acc"], marker="s", linestyle="--", label="Emulated BS=64 (accum=4)")
    axes[1].set_title("Validation Accuracy: Native vs Gradient Accumulation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("outputs/task6_grad_accum.png", dpi=150)
    print("\nSaved: outputs/task6_grad_accum.csv & outputs/task6_grad_accum.png")


if __name__ == "__main__":
    main()