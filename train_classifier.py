"""
AITX Exception Classifier
Trains an XGBoost multi-class model to predict exception codes from recon features.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

SEED = 42
MODEL_PATH = "exception_classifier.pkl"
ENCODER_PATH = "exception_label_encoder.pkl"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_all():
    dtype = {"trade_id": str, "isin": str, "cusip": str, "counterparty_bic": str,
             "asset_class": str, "currency": str, "account_code": str}
    parse_dates = ["settlement_date", "trade_date"]

    internal     = pd.read_csv("data/internal_trades.csv",    dtype=dtype, parse_dates=parse_dates)
    counterparty = pd.read_csv("data/counterparty_trades.csv", dtype=dtype, parse_dates=parse_dates)
    recon        = pd.read_csv("data/reconciliation_results.csv", dtype={"trade_id": str,
                                "exception_code": str, "status": str})
    labels       = pd.read_csv("data/break_labels.csv", dtype={"trade_id": str, "break_type": str})

    print(f"  Internal trades:        {len(internal):>7,}")
    print(f"  Counterparty trades:    {len(counterparty):>7,}")
    print(f"  Recon results:          {len(recon):>7,}")
    print(f"  Ground-truth labels:    {len(labels):>7,}")
    return internal, counterparty, recon, labels


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def build_features(internal, counterparty, recon):
    """
    Join internal vs counterparty on trade_id (left join) to compute raw
    field-comparison features, then merge in recon diff columns.
    MIS-001 trades have no counterparty match — fill binary flags with 0.
    """
    cpty_deduped = (counterparty
                    .sort_values("trade_id")
                    .drop_duplicates(subset="trade_id", keep="first")
                    [["trade_id", "isin", "cusip", "counterparty_bic", "quantity", "price"]]
                    .rename(columns={"isin": "isin_c", "cusip": "cusip_c",
                                     "counterparty_bic": "bic_c",
                                     "quantity": "qty_c", "price": "prc_c"}))

    merged = internal.merge(cpty_deduped, on="trade_id", how="left")

    # Binary field-match flags (0 when counterparty row is absent)
    merged["has_isin_match"]  = (merged["isin"]  == merged["isin_c"]).astype(float)
    merged["has_cusip_match"] = (merged["cusip"] == merged["cusip_c"]).astype(float)
    merged["has_bic_match"]   = (merged["counterparty_bic"] == merged["bic_c"]).astype(float)

    # For MIS-001 rows (no counterparty), flags are NaN → fill with 0
    for col in ("has_isin_match", "has_cusip_match", "has_bic_match"):
        merged[col] = merged[col].fillna(0)

    # Asset class encoding: FX variants=0, Equity=1, Fixed Income=2
    ac_map = {"FX_SPOT": 0, "FX_FORWARD": 0, "EQUITY": 1, "FIXED_INCOME": 2}
    merged["asset_class_enc"] = merged["asset_class"].map(ac_map).fillna(0).astype(int)

    # Trade value using internal quantity * price
    merged["trade_value"] = merged["quantity"] * merged["price"]

    # Pull in diff columns from recon results (indexed on internal trade_id)
    # MIS-002 rows in recon don't have an internal trade — exclude them
    recon_int = recon[recon["status"].isin(["MATCHED", "EXCEPTION"])].copy()
    recon_int = recon_int[["trade_id", "quantity_diff_pct", "price_diff_pct",
                            "settlement_days_diff"]].drop_duplicates(subset="trade_id")

    features = merged.merge(recon_int, on="trade_id", how="left")

    # Fill diff columns with 0 for MIS-001 (no counterparty to compare against)
    for col in ("quantity_diff_pct", "price_diff_pct", "settlement_days_diff"):
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)

    feature_cols = [
        "quantity_diff_pct",
        "price_diff_pct",
        "settlement_days_diff",
        "has_isin_match",
        "has_cusip_match",
        "has_bic_match",
        "asset_class_enc",
        "trade_value",
    ]

    return features[["trade_id"] + feature_cols], feature_cols


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
def train(features_df, labels_df, feature_cols):
    df = features_df.merge(labels_df, on="trade_id", how="inner")
    print(f"\n  Training samples:  {len(df):,}")
    print(f"  Label distribution:\n{df['break_type'].value_counts().to_string()}")

    X = df[feature_cols].values
    y_raw = df["break_type"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\n  Train / test split: {len(X_train):,} / {len(X_test):,}")

    # Class weights to handle imbalance (CLEAN >> exception classes)
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    n_classes = len(classes)
    weights = {cls: total / (n_classes * cnt) for cls, cnt in zip(classes, counts)}
    sample_weights = np.array([weights[yi] for yi in y_train])

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    target_names = le.inverse_transform(np.arange(n_classes))
    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT")
    print("=" * 65)
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Confusion matrix (exceptions only — exclude CLEAN)
    clean_idx = list(le.classes_).index("CLEAN")
    exc_mask  = y_test != clean_idx
    if exc_mask.sum() > 0:
        exc_classes = [c for c in le.classes_ if c != "CLEAN"]
        exc_indices = [list(le.classes_).index(c) for c in exc_classes]
        cm = confusion_matrix(y_test[exc_mask], y_pred[exc_mask], labels=exc_indices)
        cm_df = pd.DataFrame(cm, index=exc_classes, columns=exc_classes)
        print("Confusion matrix (exceptions only):")
        print(cm_df.to_string())
        print()

    return model, le, df


# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------
def print_feature_importance(model, feature_cols):
    print("=" * 65)
    print("FEATURE IMPORTANCE (gain)")
    print("=" * 65)
    importance = model.feature_importances_
    ranked = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    for feat, score in ranked:
        bar = "#" * int(score * 400)
        print(f"  {feat:<25}  {score:.4f}  {bar}")
    print()


# ---------------------------------------------------------------------------
# 5. Write predictions back to reconciliation_results.csv
# ---------------------------------------------------------------------------
def add_predictions(model, le, features_df, feature_cols,
                    recon_path="data/reconciliation_results.csv"):
    recon = pd.read_csv(recon_path, dtype={"trade_id": str})

    # Predict for all rows that have an internal trade (MIS-002 won't have features)
    feat = features_df.set_index("trade_id")
    X_all = feat[feature_cols].values

    probs      = model.predict_proba(X_all)
    preds      = le.inverse_transform(model.predict(X_all))
    confidence = probs.max(axis=1)

    pred_df = pd.DataFrame({
        "trade_id":              features_df["trade_id"].values,
        "predicted_exception":   preds,
        "prediction_confidence": np.round(confidence, 4),
    })

    # MIS-002 rows in recon have no internal trade — fill with "UNKNOWN"
    recon = recon.merge(pred_df, on="trade_id", how="left")
    recon["predicted_exception"]   = recon["predicted_exception"].fillna("UNKNOWN")
    recon["prediction_confidence"] = recon["prediction_confidence"].fillna(0.0)

    recon.to_csv(recon_path, index=False)
    print(f"  Predictions written to {recon_path}")

    # Quick accuracy check against engine-assigned codes
    exc = recon[recon["status"] == "EXCEPTION"].copy()
    exc = exc[exc["exception_code"] != ""]
    correct = (exc["predicted_exception"] == exc["exception_code"]).sum()
    print(f"  Predicted vs engine-assigned match rate: "
          f"{correct}/{len(exc)} ({correct/len(exc)*100:.1f}%) on exception rows")

    return recon


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("AITX EXCEPTION CLASSIFIER — TRAINING")
    print("=" * 65)

    print("\n[1] Loading data...")
    internal, counterparty, recon, labels = load_all()

    print("\n[2] Engineering features...")
    features_df, feature_cols = build_features(internal, counterparty, recon)
    print(f"  Features built for {len(features_df):,} internal trades")
    print(f"  Feature columns: {feature_cols}")

    print("\n[3] Training XGBoost classifier...")
    model, le, train_df = train(features_df, labels, feature_cols)

    print("[4] Feature importance...")
    print_feature_importance(model, feature_cols)

    print("[5] Saving model...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le,    ENCODER_PATH)
    print(f"  Model saved   : {MODEL_PATH}")
    print(f"  Encoder saved : {ENCODER_PATH}")

    print("\n[6] Writing predictions back to reconciliation_results.csv...")
    add_predictions(model, le, features_df, feature_cols)

    print("\nDone.")
