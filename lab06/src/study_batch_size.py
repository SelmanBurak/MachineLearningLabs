import os
import time
import tracemalloc
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


def get_peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    elif device.type == "mps":
        return torch.mps.driver_allocated_memory() / (1024 ** 2)
    else:
        _, peak = tracemalloc.get_traced_memory()
        return peak / (1024 ** 2)


def main():
    with open("configs/base.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    device = get_device(base_cfg["device"])
    batch_sizes = [32, 64, 128, 256, 512]
    epochs = 5
    results = []

    mlflow.set_experiment(base_cfg["experiment_name"])

    for bs in batch_sizes:
        cfg = yaml.safe_load(yaml.dump(base_cfg))
        cfg["data"]["batch_size"] = bs
        cfg["data"]["num_workers"] = 0
        cfg["training"]["epochs"] = epochs
        cfg["run_name"] = f"task4_bs_{bs}"

        set_seed(cfg["seed"])
        train_loader, val_loader = get_dataloaders(cfg)
        model = SelmanCNN(num_classes=cfg["model"]["num_classes"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, cfg)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elif device.type == "mps":
            torch.mps.empty_cache()
        else:
            tracemalloc.start()

        throughputs = []
        peak_mem_mb = 0.0
        final_val_acc = 0.0

        with mlflow.start_run(run_name=cfg["run_name"]):
            mlflow.log_params({
                "task": "task4_batch_size",
                "batch_size": bs,
                "epochs": epochs,
                "num_workers": 0,
            })

            for epoch in range(1, epochs + 1):
                t0 = time.perf_counter()
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                if device.type == "mps":
                    torch.mps.synchronize()
                epoch_time = time.perf_counter() - t0
                tp = len(train_loader.dataset) / epoch_time
                throughputs.append(tp)

                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                final_val_acc = val_acc
                peak_mem_mb = max(peak_mem_mb, get_peak_memory_mb(device))

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "throughput_samples_per_s": tp,
                    "peak_memory_mb": peak_mem_mb,
                }, step=epoch)

        if device.type == "cpu":
            tracemalloc.stop()

        avg_tp = sum(throughputs) / len(throughputs)
        print(
            f"BS={bs:3d} | Avg Throughput: {avg_tp:7.1f} samples/s | "
            f"Peak Mem: {peak_mem_mb:6.1f} MB | Final Val Acc: {final_val_acc:.2f}%"
        )
        results.append({
            "batch_size": bs,
            "throughput_samples_per_s": round(avg_tp, 1),
            "peak_memory_mb": round(peak_mem_mb, 2),
            "final_val_acc_pct": round(final_val_acc, 2),
        })

    df = pd.DataFrame(results)
    print("\n=== Task 4: Batch-Size Study (5 Epochs) ===")
    print(df.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/task4_batch_size.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(df["batch_size"], df["throughput_samples_per_s"], marker="o", color="#4C72B0")
    axes[0].set_title("Throughput vs Batch Size")
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Samples / sec")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(df["batch_size"], df["peak_memory_mb"], marker="s", color="#C44E52")
    axes[1].set_title("Peak Memory vs Batch Size")
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Peak Memory (MB)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(df["batch_size"], df["final_val_acc_pct"], marker="^", color="#55A868")
    axes[2].set_title("Final Val Acc (5 Epochs)")
    axes[2].set_xlabel("Batch Size")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("outputs/task4_batch_size.png", dpi=150)
    print("\nSaved: outputs/task4_batch_size.csv & outputs/task4_batch_size.png")


if __name__ == "__main__":
    main()