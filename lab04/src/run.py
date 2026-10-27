import logging
import os
import json
import yaml
import hashlib
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn_pipeline.pipeline_builder import build_preprocessing_pipeline
from sklearn_pipeline.audit_and_validate import audit_dataset_integrity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SelmanRunner")

def main():
    logger.info("Starting pipeline execution for Selman Burak Uygurtürk (72922)...")

    # Ingestion
    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = data.frame.copy()
    if "class" in df.columns:
        df = df.rename(columns={"class": "income"})
    rows, cols = df.shape
    logger.info(f"Ingested: {rows} rows, {cols} columns.")

    # Validation
    audit_dataset_integrity(df, "reports/validation_findings.md")

    # Clean missing symbol '?' to np.nan so SimpleImputer handles both safely
    df_clean = df.replace("?", np.nan)

    # Stratified Split 70/30 BEFORE cleaning
    y = (df_clean["income"].astype(str).str.strip().isin([">50K", ">50K."])).astype(int)
    X = df_clean.drop(columns=["income"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=72922, stratify=y
    )

    # Drop duplicates from train only
    train_combo = pd.concat([X_train, y_train], axis=1).drop_duplicates()
    y_train = train_combo["income"]
    X_train = train_combo.drop(columns=["income"])
    logger.info(f"Deduplicated Train: {len(X_train)} records. Test untouched: {len(X_test)} records.")

    # Fit Baseline WITHOUT derived features
    pipe_base = build_preprocessing_pipeline(include_derived_features=False)
    X_train_base = pipe_base.fit_transform(X_train)
    X_test_base = pipe_base.transform(X_test)

    clf_base = LogisticRegression(max_iter=1000, random_state=72922)
    clf_base.fit(X_train_base, y_train)
    preds_base = clf_base.predict(X_test_base)

    m_base = {
        "accuracy": round(float(accuracy_score(y_test, preds_base)), 4),
        "f1": round(float(f1_score(y_test, preds_base)), 4),
        "precision": round(float(precision_score(y_test, preds_base)), 4),
        "recall": round(float(recall_score(y_test, preds_base)), 4)
    }

    # Fit Pipeline WITH derived features
    pipe_feat = build_preprocessing_pipeline(include_derived_features=True)
    X_train_feat = pipe_feat.fit_transform(X_train)
    X_test_feat = pipe_feat.transform(X_test)

    clf_feat = LogisticRegression(max_iter=1000, random_state=72922)
    clf_feat.fit(X_train_feat, y_train)
    preds_feat = clf_feat.predict(X_test_feat)

    m_feat = {
        "accuracy": round(float(accuracy_score(y_test, preds_feat)), 4),
        "f1": round(float(f1_score(y_test, preds_feat)), 4),
        "precision": round(float(precision_score(y_test, preds_feat)), 4),
        "recall": round(float(recall_score(y_test, preds_feat)), 4)
    }

    joblib.dump(pipe_feat, "results/preprocessor.joblib")
    logger.info("Serialized fitted Scikit-Learn Pipeline to results/preprocessor.joblib")

    eval_out = {
        "baseline_without_features": m_base,
        "enhanced_with_features": m_feat,
        "improvement": {
            "accuracy_delta": round(m_feat["accuracy"] - m_base["accuracy"], 4),
            "f1_delta": round(m_feat["f1"] - m_base["f1"], 4)
        }
    }
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=4)

    # Hashes via record JSON bytes
    tr_hash = hashlib.sha256(X_train.to_json(orient="records").encode("utf-8")).hexdigest()
    te_hash = hashlib.sha256(X_test.to_json(orient="records").encode("utf-8")).hexdigest()

    cfg = {
        "pipeline_metadata": {
            "author": "Selman Burak Uygurtürk",
            "student_id": 72922,
            "architecture": "Scikit-Learn ColumnTransformer Pipeline",
            "seed": 72922
        },
        "data_hashes": {
            "train_record_sha256": tr_hash,
            "test_record_sha256": te_hash
        },
        "benchmark": eval_out
    }
    with open("config/pipeline_settings.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved pipeline configuration YAML.")
    print("\n[SUCCESS] Pipeline executed cleanly for Selman Burak Uygurtürk!")

if __name__ == "__main__":
    main()
