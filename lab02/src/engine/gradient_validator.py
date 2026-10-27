import numpy as np

def validate_gradients(model, X_sample, y_sample, epsilon=1e-5):
    """Validates analytical backpropagation gradients against numerical two-sided approximations."""
    model.forward(X_sample, y_sample)
    model.backward()
    
    max_error = 0.0
    for layer in [model.fc1, model.fc2, model.fc3]:
        flat_w = layer.weights.reshape(-1)
        flat_ana = layer.grad_W.reshape(-1)
        
        sample_indices = np.random.RandomState(42).choice(flat_w.size, size=min(10, flat_w.size), replace=False)
        for idx in sample_indices:
            orig = flat_w[idx]
            flat_w[idx] = orig + epsilon
            l_plus, _ = model.forward(X_sample, y_sample)
            flat_w[idx] = orig - epsilon
            l_minus, _ = model.forward(X_sample, y_sample)
            flat_w[idx] = orig
            
            num_grad = (l_plus - l_minus) / (2.0 * epsilon)
            diff = abs(flat_ana[idx] - num_grad)
            denom = max(abs(flat_ana[idx]), abs(num_grad), 1e-8)
            err = diff / denom
            if err > max_error:
                max_error = err
    return max_error
