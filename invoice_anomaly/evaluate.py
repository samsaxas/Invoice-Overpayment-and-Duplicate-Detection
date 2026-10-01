"""Evaluation: classification metrics, rule baseline and dollar-value capture."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


def classification_metrics(y_true, y_pred, score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, score)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, score)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def rule_baseline(X: pd.DataFrame) -> np.ndarray:
    """Naive hand-written AP rules - the bar an ML model must clear to be worth it."""
    return (
        (X["payment_to_invoice_ratio"] > 1.02)
        | (X["invoice_to_po_ratio"] > 1.04)
        | (X["invoice_number_seen_before"] == 1)
        | (X["nearest_prior_amount_gap"] < 0.005)
    ).astype(int).to_numpy()


def value_capture(test_df: pd.DataFrame, y_pred) -> dict:
    """Share of recoverable dollars sitting in invoices the model flagged for review.

    Recoverable value per anomaly: duplicate -> the full duplicate payment;
    overpayment -> payment minus invoice; PO price variance -> invoice minus PO amount.
    """
    flagged = np.asarray(y_pred) == 1
    total = float(test_df["recoverable_amount"].sum())
    caught = float(test_df.loc[flagged, "recoverable_amount"].sum())
    per_type = {}
    for t, g in test_df[test_df["is_anomaly"] == 1].groupby("anomaly_type"):
        f = flagged[g.index.map(test_df.index.get_loc)]
        per_type[t] = {
            "n": int(len(g)),
            "recall": round(float(f.mean()), 4),
            "recoverable_usd": round(float(g["recoverable_amount"].sum()), 2),
        }
    return {
        "total_recoverable_usd": round(total, 2),
        "flagged_recoverable_usd": round(caught, 2),
        "value_capture_pct": round(100 * caught / total, 2) if total else 0.0,
        "n_flagged_for_review": int(flagged.sum()),
        "by_type": per_type,
    }
