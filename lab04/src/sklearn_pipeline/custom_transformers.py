import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips continuous features to the 1st and 99th percentiles estimated on training data."""
    def __init__(self, lower=1.0, upper=99.0):
        self.lower = lower
        self.upper = upper
        self.thresholds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.thresholds_ = {}
        for c in X_df.columns:
            s = pd.to_numeric(X_df[c], errors="coerce")
            p_lo = np.nanpercentile(s, self.lower)
            p_hi = np.nanpercentile(s, self.upper)
            self.thresholds_[c] = (p_lo, p_hi)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for c, (p_lo, p_hi) in self.thresholds_.items():
            X_df[c] = np.clip(pd.to_numeric(X_df[c], errors="coerce"), p_lo, p_hi)
        return X_df.values

class DerivedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Engineers capital_net and hours_per_age within a Scikit-Learn transformer."""
    def __init__(self, age_idx=0, gain_idx=3, loss_idx=4, hours_idx=5):
        self.age_idx = age_idx
        self.gain_idx = gain_idx
        self.loss_idx = loss_idx
        self.hours_idx = hours_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.array(X, dtype=float)
        age = arr[:, self.age_idx]
        gain = arr[:, self.gain_idx]
        loss = arr[:, self.loss_idx]
        hours = arr[:, self.hours_idx]

        net_capital = gain - loss
        safe_age = np.maximum(age - 16.0, 1.0)
        h_per_age = hours / safe_age
        return np.column_stack([arr, net_capital, h_per_age])
