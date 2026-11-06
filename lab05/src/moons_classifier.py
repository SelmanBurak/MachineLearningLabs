"""
Hardware-Aware Training Benchmark on make_moons: Custom Graph Engine vs. PyTorch Baseline
Student Engineer: Selman Burak Uygurtürk (ID: 72922)
Network Architecture: 2 -> 8 -> 1 (ReLU hidden, Sigmoid output, BCE loss)
"""

import time
import random
import statistics
import json
import os
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from graph_scalar import GraphScalar


class PerceptronUnit:
    def __init__(self, n_inputs: int, is_relu: bool = True):
        # Xavier normal initialization
        std = (2.0 / n_inputs) ** 0.5
        self.weights = [GraphScalar(random.gauss(0.0, std)) for _ in range(n_inputs)]
        self.bias = GraphScalar(0.0)
        self.is_relu = is_relu

    def forward(self, inputs):
        acc = self.bias
        for w, x in zip(self.weights, inputs):
            acc = acc + (w * x)
        return acc.relu() if self.is_relu else acc

    def parameters(self):
        return self.weights + [self.bias]


class FullyConnectedLayer:
    def __init__(self, n_in: int, n_out: int, is_relu: bool = True):
        self.units = [PerceptronUnit(n_in, is_relu=is_relu) for _ in range(n_out)]

    def forward(self, x):
        outputs = [u.forward(x) for u in self.units]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        p = []
        for u in self.units:
            p.extend(u.parameters())
        return p


class TinyMoonsNetwork:
    def __init__(self):
        self.l1 = FullyConnectedLayer(2, 8, is_relu=True)
        self.l2 = FullyConnectedLayer(8, 1, is_relu=False)

    def forward(self, x):
        h = self.l1.forward(x)
        out = self.l2.forward(h)
        return out.sigmoid()

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

    def reset_gradients(self):
        for p in self.parameters():
            p.grad = 0.0


class PyTorchReferenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def binary_cross_entropy_cost(y_true, y_pred):
    n = len(y_true)
    losses = []
    for yt, yp in zip(y_true, y_pred):
        if yt == 1:
            term = -yp.log()
        else:
            term = -(GraphScalar(1.0) - yp).log()
        losses.append(term)
    
    total = losses[0]
    for l in losses[1:]:
        total = total + l
    return total * (1.0 / n)


def train_custom_system(X, y, epochs=35, lr=0.25):
    random.seed(72922)
    model = TinyMoonsNetwork()
    X_scalars = [[GraphScalar(v) for v in row] for row in X]
    
    t0 = time.perf_counter()
    loss_history = []
    
    for epoch in range(epochs):
        preds = [model.forward(x) for x in X_scalars]
        loss = binary_cross_entropy_cost(y, preds)
        
        model.reset_gradients()
        loss.backward()
        
        for p in model.parameters():
            p.scalar -= lr * p.grad
            
        loss_history.append(loss.scalar)
        
    duration = time.perf_counter() - t0
    acc = sum((1 if p.scalar >= 0.5 else 0) == yt for p, yt in zip(preds, y)) / len(y)
    return duration, loss_history[-1], acc


def train_pytorch_system(X, y, epochs=35, lr=0.25):
    torch.manual_seed(72922)
    model = PyTorchReferenceNet()
    crit = nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    t0 = time.perf_counter()
    loss_history = []
    
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(tx)
        loss = crit(out, ty)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
        
    duration = time.perf_counter() - t0
    with torch.no_grad():
        preds = (model(tx) >= 0.5).float()
        acc = (preds == ty).float().mean().item()
    return duration, loss_history[-1], acc


def run_benchmark():
    print("[SELMAN UYGURTURK] Executing 5-run make_moons benchmark...")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    
    c_times = []
    p_times = []
    c_losses = []
    p_losses = []
    
    for trial in range(5):
        tc, lc, _ = train_custom_system(X, y, epochs=35, lr=0.25)
        tp, lp, _ = train_pytorch_system(X, y, epochs=35, lr=0.25)
        
        c_times.append(tc * 1000.0)
        p_times.append(tp * 1000.0)
        c_losses.append(lc)
        p_losses.append(lp)
        print(f"  Trial {trial+1}/5 | Custom: {c_times[-1]:.2f} ms | PyTorch: {p_times[-1]:.2f} ms")
        
    med_custom = statistics.median(c_times)
    med_pytorch = statistics.median(p_times)
    ratio = med_custom / med_pytorch
    
    telemetry = {
        'student': 'Selman Burak Uygurtürk (72922)',
        'trials': 5,
        'custom_median_ms': med_custom,
        'pytorch_median_ms': med_pytorch,
        'slowdown_factor': ratio,
        'custom_losses': c_losses,
        'pytorch_losses': p_losses,
        'custom_times_ms': c_times,
        'pytorch_times_ms': p_times
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/moons_5runs_benchmark.json', 'w') as f:
        json.dump(telemetry, f, indent=2)
        
    print(f"\n[SUMMARY] Custom Median: {med_custom:.2f} ms | PyTorch Median: {med_pytorch:.2f} ms | Ratio: {ratio:.1f}x")
    return telemetry


if __name__ == '__main__':
    run_benchmark()
