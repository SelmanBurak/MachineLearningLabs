import time
import torch
from thop import profile as thop_profile

def profile_model(model, input_shape=(1, 1, 28, 28), iters=80):
    """Measures model parameter count, FLOPs / MACs, and microsecond-level inference latency."""
    model.eval()
    x = torch.randn(*input_shape)
    
    params = sum(p.numel() for p in model.parameters())
    macs, _ = thop_profile(model, inputs=(x,), verbose=False)
    
    with torch.no_grad():
        for _ in range(15):
            _ = model(x)
        
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        total_time = time.perf_counter() - t0
        
    latency = (total_time / iters) * 1000.0
    return {
        "parameters": params,
        "macs": int(macs),
        "mflops": (macs * 2) / 1e6,
        "latency_ms": latency
    }
