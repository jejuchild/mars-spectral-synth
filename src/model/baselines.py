"""CPU-only multi-output regression baselines for HiRISE → Mastcam-Z 11-band synthesis.

Track A only (per `context/SPECTRAL_SYNTH_DESIGN.md` §6, MARS_AUTO_OVERNIGHT.md §⑥.1).
Each baseline returns a sklearn-compatible estimator that supports `fit(X, y)` and
`predict(X)` where X is `(N, F)` and y is `(N, 11)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def make_ridge(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("est", Ridge(alpha=alpha)),
    ])


def make_random_forest(
    n_estimators: int = 80,
    max_depth: int | None = 12,
    n_jobs: int = 1,
    random_state: int = 42,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def make_xgboost(
    n_estimators: int = 100,
    max_depth: int = 6,
    n_jobs: int = 1,
    tree_method: str = "hist",
    random_state: int = 42,
) -> MultiOutputRegressor:
    from xgboost import XGBRegressor
    base = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        tree_method=tree_method,
        device="cpu",
        random_state=random_state,
        verbosity=0,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def make_mlp(
    hidden_layer_sizes: tuple[int, ...] = (64, 64),
    max_iter: int = 200,
    early_stopping: bool = True,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("est", MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )),
    ])


REGISTRY: dict[str, Any] = {
    "ridge": make_ridge,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
    "mlp": make_mlp,
}


def make_model(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}, choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


@dataclass
class TrainResult:
    name: str
    model: Any
    y_pred: np.ndarray
    y_true: np.ndarray


def fit_predict(name: str, X_train, y_train, X_test, y_test, **kwargs) -> TrainResult:
    model = make_model(name, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return TrainResult(name=name, model=model, y_pred=np.asarray(y_pred, dtype=np.float32), y_true=np.asarray(y_test, dtype=np.float32))
