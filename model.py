"""
Stage 3 - modelling and validation.

Design choices worth defending in interview:

1. The headline validation is a TEMPORAL holdout (fit on 2010-2022, test on
   2023-2025), not a random 70/30 split. Rows are repeated observations on the
   same players over time; a random split puts a player's January match in
   train and their February match in test, and the model then scores well
   partly by memorising the player. The random split is still computed and
   reported, purely to quantify how much optimism it introduces.

2. Cross-validation uses GroupKFold on player_id, for the same reason.

3. Accuracy is reported but treated as uninformative: at a ~3.6% event rate a
   model that never predicts an event scores ~96% accuracy. The metrics that
   matter for pricing are AUC (ranking), PR-AUC (ranking under imbalance) and
   Brier score / calibration (are the probabilities themselves usable as
   expected frequencies).

4. Calibration is the binding constraint. A premium is a probability times a
   cost, so a model that ranks perfectly but is systematically 30% too high is
   useless for pricing while looking fine on AUC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def make_logit(C_reg: float = 1.0) -> Pipeline:
    """L2-penalised logistic regression - the interpretable production model."""
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    C=C_reg,
                    max_iter=2000,
                    solver="lbfgs",
                    # NOTE: no class_weight balancing. Re-weighting improves
                    # recall but destroys calibration, and calibrated
                    # probabilities are the deliverable here.
                ),
            ),
        ]
    )


def make_gbm() -> Pipeline:
    """Gradient boosting challenger - tests how much the linear form gives up."""
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=4,
                    learning_rate=0.05,
                    max_iter=300,
                    l2_regularization=1.0,
                    random_state=C.RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(y_true: np.ndarray, p: np.ndarray) -> dict:
    """Full metric set. Accuracy included only to show why it misleads."""
    return {
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "auc_roc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "accuracy_at_0.5": float(accuracy_score(y_true, (p >= 0.5).astype(int))),
        "accuracy_predict_none": float(1 - np.mean(y_true)),
        "mean_pred": float(np.mean(p)),
        "obs_over_pred": float(np.mean(y_true) / np.mean(p)),
        # Decile lift is the metric that actually matters for rating: it says
        # how much more often the riskiest tenth of the book has a claim than
        # the safest tenth. A pricing model can have a modest AUC and still
        # separate rating classes usefully, and vice versa.
        "decile_lift": _decile_lift(y_true, p),
        "top_decile_rate": _decile_rate(y_true, p, top=True),
        "bottom_decile_rate": _decile_rate(y_true, p, top=False),
    }


def _decile_rate(y: np.ndarray, p: np.ndarray, top: bool) -> float:
    k = max(int(len(p) * 0.1), 1)
    order = np.argsort(p)
    idx = order[-k:] if top else order[:k]
    return float(np.mean(y[idx]))


def _decile_lift(y: np.ndarray, p: np.ndarray) -> float:
    lo = _decile_rate(y, p, top=False)
    hi = _decile_rate(y, p, top=True)
    return float(hi / lo) if lo > 0 else float("nan")


@dataclass
class SplitResult:
    name: str
    metrics: dict = field(default_factory=dict)


def temporal_split(df: pd.DataFrame):
    train = df[df["season"] <= C.TEMPORAL_SPLIT_YEAR]
    test = df[df["season"] > C.TEMPORAL_SPLIT_YEAR]
    return train, test


def fit_and_score(model: Pipeline, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    X_tr, y_tr = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET].values
    X_te, y_te = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test[TARGET].values
    model.fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]
    p_tr = model.predict_proba(X_tr)[:, 1]
    return model, evaluate(y_tr, p_tr), evaluate(y_te, p_te), p_te


def grouped_cv(model_factory, df: pd.DataFrame, n_splits: int = C.CV_FOLDS) -> pd.DataFrame:
    """5-fold CV with players held out entirely, never split across folds."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values
    groups = df["player_id"].values
    rows = []
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        m = model_factory()
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        r = evaluate(y[te], p)
        r["fold"] = k
        rows.append(r)
    return pd.DataFrame(rows)


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    """
    Odds ratios from the fitted logistic regression.

    Numeric features are standardised, so each odds ratio is the multiplicative
    effect on the odds of a non-completion event of a one-standard-deviation
    increase in that feature, holding the rest fixed. Categoricals are versus
    the dropped reference level.
    """
    prep = model.named_steps["prep"]
    names = list(NUMERIC_FEATURES)
    ohe = prep.named_transformers_["cat"]
    names += list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    coefs = model.named_steps["clf"].coef_[0]
    out = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
        }
    )
    out["abs_coef"] = out["coefficient"].abs()
    return out.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Decile reliability table: predicted vs observed frequency."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], q=bins, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(
        n=("y", "size"),
        predicted=("p", "mean"),
        observed=("y", "mean"),
    )
    g["ratio_obs_pred"] = g["observed"] / g["predicted"]
    return g.reset_index()



def vif_table(df: pd.DataFrame, sample: int = 20_000) -> pd.DataFrame:
    """
    Variance inflation factors on the design matrix.

    Run before finalising the feature set. Anything above ~10 means the
    coefficient on that feature is not separately identified, which matters
    here because the coefficients are the deliverable - they are what a
    pricing committee would interrogate.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    prep = make_preprocessor()
    X = prep.fit_transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    names = list(NUMERIC_FEATURES) + list(
        prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    if len(X) > sample:
        rng = np.random.default_rng(C.RANDOM_STATE)
        X = X[rng.choice(len(X), sample, replace=False)]
    with np.errstate(divide="ignore"):
        vals = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return pd.DataFrame({"feature": names, "vif": vals}).sort_values(
        "vif", ascending=False
    ).reset_index(drop=True)
