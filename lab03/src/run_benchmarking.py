import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from architectures.mlp_classifier import SelmanMLP
from architectures.cnn_classifier import SelmanCNN
from architectures.vit_classifier import SelmanViT
from architectures.latency_flops_suite import profile_model

def execute():
    print("[INFO] Benchmarking Selman Uygurturk Vision Models...")
    models = {
        "GELU Multilayer Perceptron": SelmanMLP(),
        "Batch-Normalized CNN": SelmanCNN(),
        "8-Head Vision Transformer": SelmanViT()
    }

    metrics = {}
    for name, m in models.items():
        res = profile_model(m)
        metrics[name] = res
        print(f"[{name}] Params: {res['parameters']:,} | MACs: {res['macs']:,} | Latency: {res['latency_ms']:.3f} ms")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "architecture_benchmarks.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[DONE] Benchmark results saved to {out_dir}")

if __name__ == "__main__":
    execute()
