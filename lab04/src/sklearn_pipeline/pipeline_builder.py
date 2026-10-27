import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_pipeline.custom_transformers import OutlierClipper, DerivedFeatureExtractor

NUMERICAL_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_COLS = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

def build_preprocessing_pipeline(include_derived_features: bool = True):
    num_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", OutlierClipper(1.0, 99.0))
    ]
    if include_derived_features:
        num_steps.append(("features", DerivedFeatureExtractor()))
    num_steps.append(("scaler", StandardScaler()))

    num_pipeline = Pipeline(num_steps)
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, NUMERICAL_COLS),
        ("cat", cat_pipeline, CATEGORICAL_COLS)
    ])
