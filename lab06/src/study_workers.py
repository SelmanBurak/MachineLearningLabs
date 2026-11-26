import os
import time
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SelmanCNN
from train import set_seed, get_device, build_optimizer


def main():
    with open("configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg["device"])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_ds = datasets.FashionMNIST(
        root=cfg["data"]["data_dir"], train=True, download=True, transform=transform
    )

    worker_counts = [0, 1, 2, 4, 8]
    results = []

    for nw in worker_counts:
        set_seed(cfg["seed"])
        loader = DataLoader(
            train_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=nw,
        )
        model = SelmanCNN(num_classes=cfg["model"]["num_classes"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, cfg)
        model.train()

        t0 = time.perf_counter()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0
        throughput = len(train_ds) / elapsed

        print(f"num_workers={nw} | Epoch Time: {elapsed:.2f}s | Throughput: {throughput:.0f} samples/s")
        results.append({
            "num_workers": nw,
            "epoch_time_s": round(elapsed, 2),
            "throughput_samples_per_s": round(throughput, 1),
        })

    df = pd.DataFrame(results)
    print("\n=== Task 3: num_workers Study ===")
    print(df.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/task3_workers.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["num_workers"], df["epoch_time_s"], marker="o", color="#C44E52", linewidth=2, label="Epoch Time (s)")
    ax1.set_xlabel("num_workers")
    ax1.set_ylabel("Epoch Time (s)", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax1.set_xticks(worker_counts)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(df["num_workers"], df["throughput_samples_per_s"], marker="s", color="#4C72B0", linewidth=2, label="Throughput (samples/s)")
    ax2.set_ylabel("Throughput (samples/s)", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")

    plt.title("DataLoader num_workers Bottleneck Study (SelmanCNN)")
    fig.tight_layout()
    plt.savefig("outputs/task3_workers.png", dpi=150)
    print("\nSaved: outputs/task3_workers.csv & outputs/task3_workers.png")


if __name__ == "__main__":
    main()
