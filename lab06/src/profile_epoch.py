import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

from model import SelmanCNN
from train import set_seed, get_device, get_dataloaders, build_optimizer


def main():
    with open("configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    train_loader, _ = get_dataloaders(cfg)

    model = SelmanCNN(num_classes=cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)

    model.train()
    activities = [ProfilerActivity.CPU]

    with profile(activities=activities, record_shapes=False) as prof:
        data_iter = iter(train_loader)
        while True:
            with record_function("data_loading"):
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    break
                images, labels = images.to(device), labels.to(device)

            with record_function("forward_pass"):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

            with record_function("backward_pass"):
                loss.backward()

            with record_function("optimizer_step"):
                optimizer.step()

            if device.type == "mps":
                torch.mps.synchronize()

    stages = ["data_loading", "forward_pass", "backward_pass", "optimizer_step"]
    times_ms = {s: 0.0 for s in stages}

    for evt in prof.key_averages():
        if evt.key in times_ms:
            times_ms[evt.key] = evt.cpu_time_total / 1000.0  # us -> ms

    total_ms = sum(times_ms.values())
    rows = []
    for s in stages:
        pct = (times_ms[s] / total_ms * 100.0) if total_ms > 0 else 0.0
        rows.append({"Stage": s, "Time (ms)": round(times_ms[s], 2), "Share (%)": round(pct, 2)})

    df = pd.DataFrame(rows)
    print("\n=== Task 2: One-Epoch Profiling Breakdown ===")
    print(df.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/task2_profile_breakdown.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(df["Stage"], df["Share (%)"], color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    plt.ylabel("Share of Epoch Time (%)")
    plt.title("One-Epoch Training Time Breakdown (SelmanCNN)")
    for i, v in enumerate(df["Share (%)"]):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.ylim(0, max(df["Share (%)"]) + 12)
    plt.tight_layout()
    plt.savefig("outputs/task2_profile_breakdown.png", dpi=150)
    print("\nSaved: outputs/task2_profile_breakdown.csv & outputs/task2_profile_breakdown.png")


if __name__ == "__main__":
    main()