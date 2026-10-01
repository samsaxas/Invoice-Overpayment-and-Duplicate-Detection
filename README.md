# Invoice Overpayment & Duplicate Detection (ML)

An end-to-end Python pipeline that flags **duplicate invoices, overpayments and PO price variances**
in accounts-payable data, then estimates how many recoverable dollars the flagged queue captures.
Built on a simulated AP ledger (6,400 invoices, 60 vendors, 6.7% anomalies) so every anomaly has a
ground-truth label *and* a ground-truth dollar value.

## Pipeline

```
data_generation.py -> features.py -> models.py -> evaluate.py -> reports/
  simulate ledger     ledger feats     IF vs RF     F1/PR-AUC     metrics.json, plots,
  + inject anomalies  + vendor z-score + rule base  + $ capture   review queue CSV
```

| Anomaly | How it is injected | Recoverable value |
|---|---|---|
| Duplicate | Earlier invoice resubmitted 1-21 days later (exact / re-formatted number / cents of drift) | full duplicate payment |
| Overpayment | Payment 0.5-60% above invoice, or paid twice | payment - invoice |
| PO price variance | Invoiced unit price 1-30% above PO price | invoice - PO amount |

**Hard negatives** keep it honest: ~1/6 of vendors bill a fixed fee on a cadence (repeated amounts are
legitimate), 35% of invoices have small PO noise, 6% are paid early with a 2% discount and 4% carry a
late fee of up to 2%.

## Features

* `payment_to_invoice_ratio`, `invoice_to_po_ratio`, `payment_to_po_ratio`, `days_to_payment`, `log_invoice_amount`
* **Nearest-prior-invoice amount gap**: relative gap to the closest earlier invoice from the same vendor within 60 days,
  plus `days_to_nearest_prior` and `near_match_count_60d` (separates real duplicates from fixed-fee vendors)
* `invoice_number_seen_before`: normalised match (case, punctuation and `-A` suffix ignored)
* **Vendor-level z-scores**: robust (median/MAD) z-scores of amount, payment ratio, PO ratio and amount gap against each vendor's own history

**Leakage controls:** ledger features only look at *earlier* invoices (tested); vendor z-scores are fitted on the
training split only (and re-fitted inside each CV fold); `recoverable_amount` / labels are never features (tested).

## Results (seed 42, stratified 80/20 split, 1,280 held-out invoices, 86 anomalies)

| Model | Precision | Recall | F1 | PR-AUC | Recoverable $ flagged |
|---|---|---|---|---|---|
| Isolation Forest (unsupervised) | 0.667 | 0.698 | 0.682 | 0.816 | 68.2% |
| Random Forest (class-weighted) | 1.000 | 1.000 | 1.000 | 1.000 | 100.0% |
| Hand-written rules | 0.190 | 0.954 | 0.317 | 0.184 | 99.8% |

Random Forest 5-fold CV F1: **0.994 ± 0.007** (the more reliable figure; a single 86-anomaly test set is small).

Takeaways:
* Isolation Forest misses most duplicates (39% recall): a duplicate looks like a *normal* invoice in feature space; it only
  becomes anomalous relative to ledger history, which a supervised model can learn.
* Rules catch nearly everything but flood reviewers (431 flags to find 82 real issues); the RF reviews 86 invoices for 86 hits.
* Value-weighted capture exceeds count recall because the largest dollars sit in duplicates.

## Caveats (read before quoting these numbers)

* The data is **simulated**. Anomalies are injected with known rules, so near-perfect supervised scores are expected
  and will not transfer to real AP data. The point is the pipeline and leakage-safe evaluation, not the headline F1.
* Isolation Forest is given the true contamination rate; in practice that is a guess.
* Metrics move with the seed and with the generator's overlap settings; re-run and quote what `reports/metrics.json` says.

## Run it

```bash
pip install -r requirements.txt
python -m invoice_anomaly.pipeline --out-dir .      # data/, reports/metrics.json, plots, review queue
python -m pytest -q                                  # tests
```

Options: `--n-invoices 6400 --n-vendors 60 --anomaly-rate 0.067 --seed 42`.

Outputs: `reports/metrics.json`, `reports/pr_curve.png`, `reports/feature_importance.png`,
`reports/flagged_for_review.csv` (held-out invoices ranked by probability x payment amount).

## Ideas to extend

Time-based split instead of random; threshold tuning on cost of review vs recovery; fuzzy vendor/number matching
(Levenshtein); gradient boosting; SHAP explanations on the review queue.
