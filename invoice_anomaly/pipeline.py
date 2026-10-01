"""End-to-end pipeline: simulate -> features -> train/benchmark -> evaluate -> reports.

Run:  python -m invoice_anomaly.pipeline --out-dir .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data_generation import generate_invoices
from .evaluate import classification_metrics, rule_baseline, value_capture
from .features import FEATURE_COLUMNS, VendorZScorer, add_ledger_features, make_feature_matrix
from .models import build_isolation_forest, build_random_forest, isolation_forest_scores


def cross_validate_rf(df: pd.DataFrame, seed: int, n_splits: int = 5) -> dict:
    """Stratified CV with vendor z-scores re-fitted inside every fold (no leakage)."""
    f1s = []
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(df, df["is_anomaly"]):
        d_tr, d_te = df.iloc[tr], df.iloc[te]
        scorer = VendorZScorer().fit(d_tr)
        rf = build_random_forest(seed).fit(make_feature_matrix(d_tr, scorer), d_tr["is_anomaly"])
        pred = rf.predict(make_feature_matrix(d_te, scorer))
        f1s.append(f1_score(d_te["is_anomaly"], pred))
    return {"f1_mean": round(float(np.mean(f1s)), 4), "f1_std": round(float(np.std(f1s)), 4),
            "folds": [round(float(x), 4) for x in f1s]}


def run(n_invoices: int, n_vendors: int, anomaly_rate: float, seed: int, out_dir: Path, verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    raw = generate_invoices(n_invoices, n_vendors, anomaly_rate, seed)
    raw.to_csv(out_dir / "data" / "invoices.csv", index=False)
    df = add_ledger_features(raw)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["is_anomaly"], random_state=seed)
    train_df, test_df = train_df.sort_index(), test_df.sort_index()
    y_tr, y_te = train_df["is_anomaly"], test_df["is_anomaly"]

    scorer = VendorZScorer().fit(train_df)  # fitted on train only
    X_tr, X_te = make_feature_matrix(train_df, scorer), make_feature_matrix(test_df, scorer)

    # 1) Isolation Forest - unsupervised (labels never used for fitting)
    iso = build_isolation_forest(contamination=float(y_tr.mean()), seed=seed).fit(X_tr)
    iso_pred = (iso.predict(X_te) == -1).astype(int)
    iso_score = isolation_forest_scores(iso, X_te)

    # 2) Class-weighted Random Forest - supervised
    rf = build_random_forest(seed).fit(X_tr, y_tr)
    rf_proba = rf.predict_proba(X_te)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)

    # 3) Hand-written rule baseline
    rule_pred = rule_baseline(X_te)

    results = {
        "dataset": {
            "n_invoices": int(len(df)), "n_vendors": int(df["vendor_id"].nunique()),
            "anomaly_rate": round(float(df["is_anomaly"].mean()), 4),
            "anomaly_counts": df["anomaly_type"].value_counts().to_dict(),
            "n_train": int(len(train_df)), "n_test": int(len(test_df)),
            "n_test_anomalies": int(y_te.sum()), "seed": seed,
        },
        "isolation_forest": {**classification_metrics(y_te, iso_pred, iso_score),
                             "value": value_capture(test_df, iso_pred)},
        "random_forest": {**classification_metrics(y_te, rf_pred, rf_proba),
                          "value": value_capture(test_df, rf_pred),
                          "cv": cross_validate_rf(df, seed)},
        "rule_baseline": {**classification_metrics(y_te, rule_pred, rule_pred),
                          "value": value_capture(test_df, rule_pred)},
    }
    with open(out_dir / "reports" / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _save_plots(out_dir / "reports", y_te, iso_score, rf_proba, rf)
    _save_review_queue(out_dir / "reports" / "flagged_for_review.csv", test_df, rf_proba, rf_pred)
    if verbose:
        _print_summary(results)
    return results


def _save_plots(rep: Path, y_te, iso_score, rf_proba, rf) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in [("Random Forest", rf_proba), ("Isolation Forest", iso_score)]:
        p, r, _ = precision_recall_curve(y_te, s)
        ax.plot(r, p, label=name)
    ax.axhline(y_te.mean(), ls="--", c="grey", label=f"Base rate ({y_te.mean():.1%})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (held-out set)")
    ax.legend(); fig.tight_layout(); fig.savefig(rep / "pr_curve.png", dpi=150); plt.close(fig)

    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    imp.plot.barh(ax=ax, color="#3b6ea5")
    ax.set(title="Random Forest feature importance", xlabel="Mean decrease in impurity")
    fig.tight_layout(); fig.savefig(rep / "feature_importance.png", dpi=150); plt.close(fig)


def _save_review_queue(path: Path, test_df: pd.DataFrame, proba, pred) -> None:
    q = test_df.assign(anomaly_probability=proba.round(4))[pred == 1]
    q = q.assign(review_priority=(q["anomaly_probability"] * q["payment_amount"]).round(2))
    cols = ["invoice_id", "invoice_number", "vendor_id", "invoice_date", "invoice_amount",
            "payment_amount", "po_amount", "anomaly_probability", "review_priority"]
    q[cols].sort_values("review_priority", ascending=False).to_csv(path, index=False)


def _print_summary(r: dict) -> None:
    d = r["dataset"]
    print(f"Ledger: {d['n_invoices']:,} invoices, {d['n_vendors']} vendors, "
          f"{d['anomaly_rate']:.1%} anomalies | held-out: {d['n_test']:,} ({d['n_test_anomalies']} anomalies)")
    print(f"{'Model':<18}{'Prec':>7}{'Rec':>7}{'F1':>7}{'PR-AUC':>9}{'$ captured':>12}")
    for key, label in [("isolation_forest", "Isolation Forest"), ("random_forest", "Random Forest"),
                       ("rule_baseline", "Rule baseline")]:
        m = r[key]
        print(f"{label:<18}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}"
              f"{m['pr_auc']:>9.3f}{m['value']['value_capture_pct']:>11.1f}%")
    cv = r["random_forest"]["cv"]
    print(f"Random Forest 5-fold CV F1: {cv['f1_mean']:.3f} +/- {cv['f1_std']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-invoices", type=int, default=6400)
    ap.add_argument("--n-vendors", type=int, default=60)
    ap.add_argument("--anomaly-rate", type=float, default=0.067)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    a = ap.parse_args()
    run(a.n_invoices, a.n_vendors, a.anomaly_rate, a.seed, a.out_dir)


if __name__ == "__main__":
    main()
