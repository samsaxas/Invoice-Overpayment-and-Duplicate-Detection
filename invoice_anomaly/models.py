"""Model builders: unsupervised Isolation Forest vs class-weighted Random Forest."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier


def build_isolation_forest(contamination: float, seed: int = 42) -> IsolationForest:
    """Unsupervised baseline. ``contamination`` encodes the expected anomaly rate."""
    return IsolationForest(n_estimators=300, contamination=contamination, random_state=seed, n_jobs=-1)


def build_random_forest(seed: int = 42) -> RandomForestClassifier:
    """Supervised model; ``balanced_subsample`` re-weights the ~7% minority class per tree."""
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def isolation_forest_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous."""
    return -model.decision_function(X)
