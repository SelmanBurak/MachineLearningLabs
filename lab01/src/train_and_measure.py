import os
import time
import psutil
import joblib
import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("[INFO] Executing Selman Burak Uygurturk's Lab 01 pipeline...")
