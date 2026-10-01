import numpy as np
import pandas as pd

from invoice_anomaly.data_generation import generate_invoices
from invoice_anomaly.features import (
    FEATURE_COLUMNS, VendorZScorer, add_ledger_features, make_feature_matrix, normalize_invoice_number,
)
from invoice_anomaly.evaluate import value_capture
from invoice_anomaly.pipeline import run


def test_generator_shape_and_rate():
    df = generate_invoices(n_invoices=1000, n_vendors=20, anomaly_rate=0.07, seed=1)
    assert len(df) == 1000
    assert df["vendor_id"].nunique() <= 20
    assert abs(df["is_anomaly"].mean() - 0.07) < 0.005
    assert df["invoice_date"].is_monotonic_increasing
    assert set(df["anomaly_type"]) == {"none", "duplicate", "overpayment", "po_price_variance"}
    assert (df.loc[df.is_anomaly == 0, "recoverable_amount"] == 0).all()


def test_generator_is_reproducible():
    a = generate_invoices(500, 10, seed=7)
    b = generate_invoices(500, 10, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_normalize_invoice_number():
    assert normalize_invoice_number("V001-00012") == normalize_invoice_number("v00100012")
    assert normalize_invoice_number("V001-00012A") == normalize_invoice_number("V001-00012")


def _toy():
    return pd.DataFrame({
        "invoice_number": ["V1-1", "V1-2", "v1-1", "V2-1"],
        "vendor_id": ["V1", "V1", "V1", "V2"],
        "invoice_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-10"]),
        "payment_date": pd.to_datetime(["2024-01-20", "2024-01-25", "2024-02-01", "2024-02-01"]),
        "invoice_amount": [100.0, 250.0, 100.0, 100.0],
        "po_amount": [100.0, 250.0, 100.0, 100.0],
        "payment_amount": [100.0, 250.0, 100.0, 100.0],
    })


def test_nearest_prior_gap_and_number_seen():
    f = add_ledger_features(_toy())
    assert f.loc[0, "nearest_prior_amount_gap"] == 1.0          # no prior invoice
    assert f.loc[2, "nearest_prior_amount_gap"] == 0.0          # re-submission of invoice 0
    assert f.loc[2, "days_to_nearest_prior"] == 9
    assert f.loc[2, "invoice_number_seen_before"] == 1
    assert f.loc[3, "nearest_prior_amount_gap"] == 1.0          # other vendor's history is not used


def test_features_only_use_prior_invoices():
    f1 = add_ledger_features(_toy())
    extra = pd.concat([_toy(), _toy().assign(invoice_date=pd.Timestamp("2024-03-01") + pd.to_timedelta([0, 1, 2, 3], "D"),
                                             payment_date=pd.Timestamp("2024-04-01"))], ignore_index=True)
    f2 = add_ledger_features(extra)
    cols = ["nearest_prior_amount_gap", "days_to_nearest_prior", "invoice_number_seen_before"]
    pd.testing.assert_frame_equal(f1[cols], f2.loc[:3, cols])  # future rows never change past features


def test_zscorer_fit_on_train_only_and_unseen_vendor():
    df = add_ledger_features(generate_invoices(800, 12, seed=3))
    scorer = VendorZScorer().fit(df[df.vendor_id != "V001"])
    X = make_feature_matrix(df, scorer)           # V001 is unseen -> global fallback, no NaNs
    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isna().any().any()
    assert X.abs().max().max() < 1e6


def test_no_label_columns_in_features():
    banned = {"is_anomaly", "anomaly_type", "recoverable_amount"}
    assert banned.isdisjoint(FEATURE_COLUMNS)


def test_value_capture_math():
    t = pd.DataFrame({"is_anomaly": [1, 1, 0], "anomaly_type": ["duplicate", "overpayment", "none"],
                      "recoverable_amount": [100.0, 50.0, 0.0]})
    v = value_capture(t, np.array([1, 0, 0]))
    assert v["value_capture_pct"] == round(100 * 100 / 150, 2)


def test_end_to_end_smoke(tmp_path):
    r = run(1200, 20, 0.07, 5, tmp_path, verbose=False)
    assert r["random_forest"]["pr_auc"] > r["isolation_forest"]["pr_auc"]
    assert (tmp_path / "reports" / "metrics.json").exists()
    assert (tmp_path / "reports" / "flagged_for_review.csv").exists()
