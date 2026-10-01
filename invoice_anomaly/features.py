"""Feature engineering for invoice anomaly detection.

Two kinds of features:

1. **Ledger features** - computed per invoice from the row itself and from *earlier* invoices
   of the same vendor only (no labels, no future data), so they can be built once on the
   whole ledger without leakage.
2. **Vendor z-scores** - robust (median/MAD) z-scores against each vendor's own history.
   These are *fitted on the training split only* (``VendorZScorer``) and then applied to
   held-out data, so test-set statistics never leak into training.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

LOOKBACK_DAYS = 60
NEAR_MATCH_TOL = 0.005  # amounts within 0.5% count as a "near match"

ZSCORE_SOURCES = {  # column -> floor on the scaled MAD (avoids blow-ups for constant vendors)
    "log_invoice_amount": 0.05,
    "payment_to_invoice_ratio": 0.005,
    "invoice_to_po_ratio": 0.005,
    "nearest_prior_amount_gap": 0.05,
}

LEDGER_FEATURES = [
    "payment_to_invoice_ratio",
    "invoice_to_po_ratio",
    "payment_to_po_ratio",
    "days_to_payment",
    "log_invoice_amount",
    "nearest_prior_amount_gap",
    "days_to_nearest_prior",
    "near_match_count_60d",
    "invoice_number_seen_before",
]
FEATURE_COLUMNS = LEDGER_FEATURES + [f"z_{c}" for c in ZSCORE_SOURCES]


def normalize_invoice_number(num: str) -> str:
    """Upper-case, drop punctuation and trailing letter suffixes (``-A`` re-submissions)."""
    return re.sub(r"[A-Z]+$", "", re.sub(r"[^A-Z0-9]", "", str(num).upper()))


def add_ledger_features(df: pd.DataFrame, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Add per-invoice and vendor-history features. ``df`` must be sorted by invoice_date."""
    df = df.reset_index(drop=True).copy()
    if not df["invoice_date"].is_monotonic_increasing:
        raise ValueError("df must be sorted by invoice_date")

    df["payment_to_invoice_ratio"] = df["payment_amount"] / df["invoice_amount"]
    df["invoice_to_po_ratio"] = df["invoice_amount"] / df["po_amount"]
    df["payment_to_po_ratio"] = df["payment_amount"] / df["po_amount"]
    df["days_to_payment"] = (df["payment_date"] - df["invoice_date"]).dt.days
    df["log_invoice_amount"] = np.log1p(df["invoice_amount"])

    n = len(df)
    gap = np.ones(n)  # relative amount gap to the closest prior invoice (1.0 = nothing close)
    days = np.full(n, float(lookback_days))
    near_count = np.zeros(n)
    seen = np.zeros(n)

    amounts = df["invoice_amount"].to_numpy()
    day_num = (df["invoice_date"] - df["invoice_date"].min()).dt.days.to_numpy()
    numbers = df["invoice_number"].map(normalize_invoice_number).to_numpy()

    for _, pos in df.groupby("vendor_id").indices.items():  # positions, ascending by date
        seen_numbers: set[str] = set()
        for k, i in enumerate(pos):
            seen[i] = float(numbers[i] in seen_numbers)
            seen_numbers.add(numbers[i])
            prior = pos[:k]
            if len(prior) == 0:
                continue
            prior = prior[day_num[i] - day_num[prior] <= lookback_days]
            if len(prior) == 0:
                continue
            rel = np.abs(amounts[prior] - amounts[i]) / max(amounts[i], 1e-9)
            best = np.flatnonzero(rel == rel.min())[-1]  # most recent among ties
            gap[i] = min(rel[best], 1.0)
            days[i] = day_num[i] - day_num[prior[best]]
            near_count[i] = (rel < NEAR_MATCH_TOL).sum()

    df["nearest_prior_amount_gap"] = gap
    df["days_to_nearest_prior"] = days
    df["near_match_count_60d"] = near_count
    df["invoice_number_seen_before"] = seen
    return df


class VendorZScorer:
    """Robust per-vendor z-scores, fitted on training data only."""

    def __init__(self, sources: dict[str, float] | None = None, clip: float = 50.0):
        self.sources = sources or ZSCORE_SOURCES
        self.clip = clip
        self.median_: dict[str, pd.Series] = {}
        self.mad_: dict[str, pd.Series] = {}
        self.global_: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "VendorZScorer":
        for col in self.sources:
            med = df.groupby("vendor_id")[col].median()
            dev = (df[col] - df["vendor_id"].map(med)).abs()
            mad = dev.groupby(df["vendor_id"]).median() * 1.4826
            self.median_[col], self.mad_[col] = med, mad
            g_med = float(df[col].median())
            self.global_[col] = (g_med, float((df[col] - g_med).abs().median() * 1.4826))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col, floor in self.sources.items():
            g_med, g_mad = self.global_[col]  # unseen vendors fall back to global stats
            med = df["vendor_id"].map(self.median_[col]).fillna(g_med)
            mad = df["vendor_id"].map(self.mad_[col]).fillna(g_mad)
            z = (df[col] - med) / np.maximum(mad, floor)
            out[f"z_{col}"] = z.clip(-self.clip, self.clip)
        return pd.DataFrame(out, index=df.index)


def make_feature_matrix(df: pd.DataFrame, scorer: VendorZScorer) -> pd.DataFrame:
    """Combine ledger features with fitted vendor z-scores (columns = ``FEATURE_COLUMNS``)."""
    X = pd.concat([df[LEDGER_FEATURES], scorer.transform(df)], axis=1)
    return X[FEATURE_COLUMNS]
