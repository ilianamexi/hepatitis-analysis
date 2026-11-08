"""Leakage checks on synthetic data (no download needed)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hepatitis_analysis as ha  # noqa: E402


def test_imputer_uses_training_rows_only():
    pipe = ha.models()["Logistic regression"]
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [0.0, 1.0, 0.0, 1.0]})
    y = np.array([0, 1, 0, 1])
    pipe.fit(X.iloc[:3], y[:3])
    # median of the 3 training rows only
    assert pipe.named_steps["impute"].statistics_[0] == 2.0


def test_smote_only_resamples_inside_fit():
    pipe = ha.models()["Random forest + SMOTE"]
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    y = np.array([0] * 34 + [1] * 6)
    pipe.fit(X, y)
    # predicting does not change the number of rows (no resampling at test time)
    assert len(pipe.predict(X)) == 40


def test_stage_mapping_excludes_suspect_donors():
    assert "Suspect donor" not in ha.STAGE
    assert ha.STAGE == {"Blood donor": 0, "Hepatitis": 1, "Fibrosis": 2, "Cirrhosis": 3}
