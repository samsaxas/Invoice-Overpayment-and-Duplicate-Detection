"""Simulate an accounts-payable ledger with injected, labelled anomalies.

Three anomaly families are injected:

* ``duplicate``          - the same invoice submitted (and paid) again a few days later,
                           sometimes with a re-formatted invoice number or a few cents of drift.
* ``overpayment``        - cash paid out exceeds the invoice (0.5-60%, or paid twice).
* ``po_price_variance``  - the invoiced unit price is 1-30% above the purchase-order price.

Realistic *hard negatives* keep the problem from being trivial: ~1/6 of vendors bill a
fixed fee on a regular cadence (repeated amounts are legitimate), some invoices carry small
invoice-vs-PO noise, some are paid early with a 2% discount, and some carry a small late fee.

``recoverable_amount`` is the ground-truth dollar value that could be clawed back. It is used
for *evaluation only* and must never be used as a model feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANOMALY_SPLIT = {"duplicate": 0.40, "overpayment": 0.30, "po_price_variance": 0.30}
START_DATE = pd.Timestamp("2024-01-01")
N_DAYS = 730

LEDGER_COLUMNS = [
    "invoice_id", "invoice_number", "vendor_id", "po_id", "invoice_date", "payment_date",
    "quantity", "po_unit_price", "invoice_unit_price", "po_amount", "invoice_amount",
    "payment_amount", "anomaly_type", "is_anomaly", "recoverable_amount",
]


def _make_vendors(n_vendors: int, rng: np.random.Generator) -> pd.DataFrame:
    recurring = np.zeros(n_vendors, dtype=bool)
    recurring[rng.choice(n_vendors, size=max(1, n_vendors // 6), replace=False)] = True
    return pd.DataFrame(
        {
            "vendor_id": [f"V{i:03d}" for i in range(1, n_vendors + 1)],
            "price_scale": np.exp(rng.normal(5.5, 1.0, n_vendors)),
            "weight": rng.dirichlet(np.full(n_vendors, 1.5)),
            "recurring": recurring,
        }
    )


def _clean_rows(vendors: pd.DataFrame, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate ``n_rows`` legitimate invoices (with benign noise)."""
    counts = rng.multinomial(n_rows, vendors["weight"].to_numpy())
    frames = []
    for v, cnt in zip(vendors.itertuples(index=False), counts):
        if cnt == 0:
            continue
        if v.recurring:  # fixed-fee vendor on a regular cadence
            spacing = N_DAYS / cnt
            offsets = np.arange(cnt) * spacing + rng.integers(-1, 2, cnt)
            offsets = np.clip(offsets, 0, N_DAYS - 1).astype(int)
            qty = np.ones(cnt, dtype=int)
            po_price = np.full(cnt, round(v.price_scale * rng.uniform(2, 6), 2))
        else:
            offsets = rng.integers(0, N_DAYS, cnt)
            qty = 1 + rng.poisson(6, cnt)
            po_price = np.round(v.price_scale * np.exp(rng.normal(0, 0.3, cnt)), 2)

        # benign invoice-vs-PO noise on ~35% of invoices (+/-0.6% std)
        noise = np.where(rng.random(cnt) < 0.35, rng.normal(0, 0.006, cnt), 0.0)
        inv_price = np.round(po_price * (1 + noise), 2)
        invoice_amount = np.round(qty * inv_price, 2)

        # benign payment noise: 6% early-pay discount, 4% small late fee
        u = rng.random(cnt)
        ratio = np.ones(cnt)
        ratio[u < 0.06] = 0.98
        late = (u >= 0.06) & (u < 0.10)
        ratio[late] = rng.uniform(1.0, 1.02, late.sum())

        invoice_date = START_DATE + pd.to_timedelta(offsets, unit="D")
        frames.append(
            pd.DataFrame(
                {
                    "vendor_id": v.vendor_id,
                    "_recurring": bool(v.recurring),
                    "invoice_date": invoice_date,
                    "payment_date": invoice_date + pd.to_timedelta(rng.integers(10, 46, cnt), unit="D"),
                    "quantity": qty,
                    "po_unit_price": po_price,
                    "invoice_unit_price": inv_price,
                    "po_amount": np.round(qty * po_price, 2),
                    "invoice_amount": invoice_amount,
                    "payment_amount": np.round(invoice_amount * ratio, 2),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["anomaly_type"] = "none"
    df["recoverable_amount"] = 0.0
    return df


def _variant_invoice_number(num: str, rng: np.random.Generator) -> str:
    """Re-format an invoice number the way a duplicate submission often is."""
    choice = rng.integers(0, 4)
    if choice == 0:
        return num  # exact resubmission
    if choice == 1:
        return num.replace("-", "")
    if choice == 2:
        return num.lower()
    return num + "A"


def generate_invoices(
    n_invoices: int = 6400,
    n_vendors: int = 60,
    anomaly_rate: float = 0.067,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a date-sorted AP ledger with ``n_invoices`` rows and labelled anomalies."""
    rng = np.random.default_rng(seed)
    n_anom = int(round(anomaly_rate * n_invoices))
    n_dup = int(round(ANOMALY_SPLIT["duplicate"] * n_anom))
    n_over = int(round(ANOMALY_SPLIT["overpayment"] * n_anom))
    n_var = n_anom - n_dup - n_over

    vendors = _make_vendors(n_vendors, rng)
    df = _clean_rows(vendors, n_invoices - n_dup, rng)
    df = df.sort_values("invoice_date", kind="mergesort").reset_index(drop=True)
    df["invoice_number"] = (
        df["vendor_id"] + "-" + (df.groupby("vendor_id").cumcount() + 1).astype(str).str.zfill(5)
    )
    df["po_id"] = [f"PO-{i:06d}" for i in rng.permutation(len(df)) + 1]

    order = rng.permutation(len(df))
    over_idx = order[:n_over]
    var_idx = order[n_over : n_over + n_var]

    # --- overpayments: pay more than invoiced (3-60%, or the invoice is paid twice) ---
    factor = np.where(rng.random(n_over) < 0.20, 2.0, rng.uniform(1.005, 1.60, n_over))
    inv = df.loc[over_idx, "invoice_amount"].to_numpy()
    paid = np.round(inv * factor, 2)
    df.loc[over_idx, "payment_amount"] = paid
    df.loc[over_idx, "recoverable_amount"] = np.round(paid - inv, 2)
    df.loc[over_idx, "anomaly_type"] = "overpayment"

    # --- PO price variance: invoiced price 1-30% above PO price, paid in full ---
    markup = rng.uniform(1.01, 1.30, n_var)
    po_price = df.loc[var_idx, "po_unit_price"].to_numpy()
    qty = df.loc[var_idx, "quantity"].to_numpy()
    inv_price = np.round(po_price * markup, 2)
    inv_amt = np.round(qty * inv_price, 2)
    df.loc[var_idx, "invoice_unit_price"] = inv_price
    df.loc[var_idx, "invoice_amount"] = inv_amt
    df.loc[var_idx, "payment_amount"] = inv_amt
    df.loc[var_idx, "recoverable_amount"] = np.round(inv_amt - df.loc[var_idx, "po_amount"].to_numpy(), 2)
    df.loc[var_idx, "anomaly_type"] = "po_price_variance"

    # --- duplicates: resubmit an earlier clean, non-recurring invoice 1-21 days later ---
    candidates = df.index[(df["anomaly_type"] == "none") & (~df["_recurring"])].to_numpy()
    src_idx = rng.choice(candidates, size=n_dup, replace=len(candidates) < n_dup)
    dups = df.loc[src_idx].copy().reset_index(drop=True)
    dups["invoice_number"] = [_variant_invoice_number(n, rng) for n in dups["invoice_number"]]
    dups["invoice_date"] = dups["invoice_date"] + pd.to_timedelta(rng.integers(1, 22, n_dup), unit="D")
    dups["payment_date"] = dups["invoice_date"] + pd.to_timedelta(rng.integers(10, 46, n_dup), unit="D")
    drift = np.where(rng.random(n_dup) < 0.15, np.round(rng.uniform(-0.5, 0.5, n_dup), 2), 0.0)
    dups["invoice_amount"] = np.round(dups["invoice_amount"] + drift, 2)
    dups["payment_amount"] = dups["invoice_amount"]
    dups["recoverable_amount"] = dups["payment_amount"]
    dups["anomaly_type"] = "duplicate"

    out = pd.concat([df, dups], ignore_index=True)
    out = out.sort_values("invoice_date", kind="mergesort").reset_index(drop=True)
    out["invoice_id"] = [f"INV-{i:06d}" for i in range(1, len(out) + 1)]
    out["is_anomaly"] = (out["anomaly_type"] != "none").astype(int)
    return out[LEDGER_COLUMNS]
