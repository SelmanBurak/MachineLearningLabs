"""
Gradient Verification Suite: Custom Graph Engine vs. PyTorch Autograd & Central Finite Differences
Student Engineer: Selman Burak Uygurtürk (ID: 72922)
Formula: ((a * b + c) ** 2).relu().sum()
"""

import json
import os
import torch
from graph_scalar import GraphScalar


def forward_pass(a_nodes, b_nodes, c_nodes):
    terms = []
    for a, b, c in zip(a_nodes, b_nodes, c_nodes):
        t = ((a * b + c) ** 2).relu()
        terms.append(t)
    
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def central_finite_diff(a_vals, b_vals, c_vals, param, idx, h=1e-7):
    def f(a, b, c):
        return sum(max(0.0, (ai * bi + ci) ** 2) for ai, bi, ci in zip(a, b, c))

    ap, am = list(a_vals), list(a_vals)
    bp, bm = list(b_vals), list(b_vals)
    cp, cm = list(c_vals), list(c_vals)

    if param == 'a':
        ap[idx] += h
        am[idx] -= h
    elif param == 'b':
        bp[idx] += h
        bm[idx] -= h
    else:
        cp[idx] += h
        cm[idx] -= h

    return (f(ap, bp, cp) - f(am, bm, cm)) / (2.0 * h)


def run_verification():
    print("[SELMAN UYGURTURK] Running autograd verification suite...")
    a_in = [1.35, -2.25, 0.60, -1.50]
    b_in = [-2.10, 1.40, 3.50, 0.75]
    c_in = [1.20, -0.70, -1.90, 2.50]

    # Custom GraphScalar Pass
    a_nodes = [GraphScalar(x, symbol=f'a_{i}') for i, x in enumerate(a_in)]
    b_nodes = [GraphScalar(x, symbol=f'b_{i}') for i, x in enumerate(b_in)]
    c_nodes = [GraphScalar(x, symbol=f'c_{i}') for i, x in enumerate(c_in)]

    total_loss = forward_pass(a_nodes, b_nodes, c_nodes)
    total_loss.backward()

    # PyTorch Autograd Pass
    ta = torch.tensor(a_in, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b_in, dtype=torch.float64, requires_grad=True)
    tc = torch.tensor(c_in, dtype=torch.float64, requires_grad=True)

    t_loss = (((ta * tb + tc) ** 2).relu()).sum()
    t_loss.backward()

    # Finite Differences
    max_torch_err = 0.0
    max_fd_err = 0.0
    details = []

    for name, nodes, t_ref in [('a', a_nodes, ta), ('b', b_nodes, tb), ('c', c_nodes, tc)]:
        for i in range(len(a_in)):
            g_cust = nodes[i].grad
            g_torch = t_ref.grad[i].item()
            g_fd = central_finite_diff(a_in, b_in, c_in, name, i)

            d_torch = abs(g_cust - g_torch)
            d_fd = abs(g_cust - g_fd)

            max_torch_err = max(max_torch_err, d_torch)
            max_fd_err = max(max_fd_err, d_fd)

            details.append({
                'param': f"{name}_{i}",
                'custom_grad': g_cust,
                'torch_grad': g_torch,
                'fd_grad': g_fd,
                'diff_torch': d_torch,
                'diff_fd': d_fd
            })

    passed = bool(max_torch_err < 1e-6)
    print(f"  Max Diff vs PyTorch: {max_torch_err:.2e}")
    print(f"  Max Diff vs Finite Diff: {max_fd_err:.2e}")
    print(f"  Status: {'PASSED (< 1e-6)' if passed else 'FAILED'}")

    res = {
        'student': 'Selman Burak Uygurtürk (72922)',
        'target_expression': '((a * b + c) ** 2).relu().sum()',
        'forward_loss_custom': total_loss.scalar,
        'forward_loss_torch': t_loss.item(),
        'max_torch_error': max_torch_err,
        'max_fd_error': max_fd_err,
        'threshold': 1e-6,
        'passed': passed,
        'audit': details
    }

    os.makedirs('results', exist_ok=True)
    with open('results/verification_report.json', 'w') as f:
        json.dump(res, f, indent=2)

    return res


if __name__ == '__main__':
    run_verification()
