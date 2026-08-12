"""
Stage 4 - actuarial pricing.

Product being priced: a one-season indemnity policy for a professional tennis
player, paying a fixed benefit each time the player fails to complete or take
the court for a scheduled tour match.

Structure:

    Annual pure premium = E[N] x E[X]

    E[N] = expected number of events in a season
         = (expected matches contested) x (per-match event probability)
    E[X] = expected cost per event

The exposure measure is the match, not the player-year. This matters: a top-50
player contests roughly twice as many matches as a player ranked 150-250, so
charging both the same annual premium for the same per-match risk would be
mispricing by a factor of two. Exposure is the first thing an actuary asks
about and the modelling frame is built around it.

Severity is NOT flat. The brief's £2,000 average is retained as the portfolio
mean, but it is decomposed into a minor/major mix using the observed
probability that a non-completion is followed by an extended absence. A flat
severity would make the entire premium table a rescaling of the frequency
table, which tests nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Severity model
# --------------------------------------------------------------------------
def fit_severity(df: pd.DataFrame, mean_cost: float = C.MEAN_CLAIM_COST_GBP) -> dict:
    """
    Two-point severity mix calibrated to the assumed portfolio mean.

    We observe, for events in the data, whether the player was then absent for
    at least LAYOFF_DAYS. That split is a genuine data-driven quantity. What is
    NOT data-driven is the monetary cost of either category - no public source
    prices tennis injury claims, so the ratio between major and minor cost is
    an assumption, set at 5:1 and stress-tested in sensitivity_table().
    """
    ev = df[(df["injury_event"] == 1) & df["layoff_event"].notna()]
    p_major = float(ev["layoff_event"].mean()) if len(ev) else 0.25

    ratio = 5.0  # cost_major / cost_minor - ASSUMPTION
    # mean = p_major*ratio*c_minor + (1-p_major)*c_minor  ->  solve c_minor
    c_minor = mean_cost / (p_major * ratio + (1 - p_major))
    c_major = ratio * c_minor

    # Second moment, needed for the risk loading and for process variance.
    ex2 = p_major * c_major**2 + (1 - p_major) * c_minor**2
    var = ex2 - mean_cost**2

    return {
        "p_major": p_major,
        "cost_minor": c_minor,
        "cost_major": c_major,
        "mean_severity": mean_cost,
        "var_severity": var,
        "cv_severity": np.sqrt(var) / mean_cost,
        "n_events_used": int(len(ev)),
    }


# --------------------------------------------------------------------------
# Premium calculation
# --------------------------------------------------------------------------
def pure_premium(
    freq_per_match: np.ndarray,
    matches_per_season: np.ndarray | float,
    mean_severity: float = C.MEAN_CLAIM_COST_GBP,
) -> np.ndarray:
    """Expected annual loss before any loading."""
    return np.asarray(freq_per_match) * np.asarray(matches_per_season) * mean_severity


def office_premium(
    pp: np.ndarray,
    expense_loading: float = C.EXPENSE_LOADING,
    risk_loading_sd: float = 0.0,
    sd: np.ndarray | None = None,
) -> np.ndarray:
    """
    Gross premium = pure premium, loaded.

    Two loadings are available and they do different jobs:
      - expense_loading is a proportional margin for expenses and profit;
      - risk_loading_sd charges for volatility (a standard-deviation
        principle), which is what makes a low-frequency, high-variance risk
        priced defensibly rather than at expected value.
    """
    pp = np.asarray(pp, dtype=float)
    loaded = pp * (1 + expense_loading)
    if risk_loading_sd and sd is not None:
        loaded = loaded + risk_loading_sd * np.asarray(sd, dtype=float)
    return loaded


def aggregate_sd(
    freq_per_match: np.ndarray,
    matches_per_season: np.ndarray | float,
    sev: dict,
) -> np.ndarray:
    """
    Standard deviation of annual aggregate loss under a compound Poisson model.

    Var(S) = E[N] * E[X^2]   for N ~ Poisson.

    Poisson is an approximation: a player who gets injured is more likely to be
    injured again (the model's prior_events features pick some of this up, but
    residual contagion would make the true distribution over-dispersed and this
    figure an understatement). Flagged in the limitations section.
    """
    en = np.asarray(freq_per_match) * np.asarray(matches_per_season)
    ex2 = sev["var_severity"] + sev["mean_severity"] ** 2
    return np.sqrt(en * ex2)


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
def logit_parameter_covariance(model, X_design: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Asymptotic covariance of the fitted logistic coefficients, (X'WX)^-1.

    sklearn does not expose standard errors, so we reconstruct them from the
    Fisher information at the fitted values. This gives PARAMETER uncertainty
    on the premium - the "how well do we know the rate" component, which is
    distinct from process variance ("how volatile is one player's year").
    """
    W = p * (1 - p)
    XtWX = X_design.T @ (X_design * W[:, None])
    ridge = 1e-8 * np.eye(XtWX.shape[0])
    return np.linalg.pinv(XtWX + ridge)


def premium_interval_analytic(
    eta: np.ndarray,
    se_eta: np.ndarray,
    matches_per_season: np.ndarray | float,
    mean_severity: float,
    z: float = 1.96,
) -> pd.DataFrame:
    """95% interval on the pure premium from parameter uncertainty alone."""
    lo_p = 1 / (1 + np.exp(-(eta - z * se_eta)))
    hi_p = 1 / (1 + np.exp(-(eta + z * se_eta)))
    mid_p = 1 / (1 + np.exp(-eta))
    return pd.DataFrame(
        {
            "freq_lo": lo_p,
            "freq_mid": mid_p,
            "freq_hi": hi_p,
            "pp_lo": pure_premium(lo_p, matches_per_season, mean_severity),
            "pp_mid": pure_premium(mid_p, matches_per_season, mean_severity),
            "pp_hi": pure_premium(hi_p, matches_per_season, mean_severity),
        }
    )


def bootstrap_portfolio(
    df: pd.DataFrame,
    freq_col: str,
    matches_col: str,
    sev: dict,
    n_boot: int = C.BOOTSTRAP_N,
    seed: int = C.RANDOM_STATE,
) -> dict:
    """
    Resample PLAYERS (not rows) to get an interval on the portfolio-average
    premium. Resampling rows would understate uncertainty because a player's
    matches are correlated with each other.
    """
    rng = np.random.default_rng(seed)
    players = df["player_id"].unique()
    means = []
    idx_by_player = {p: g.index.values for p, g in df.groupby("player_id")}
    for _ in range(n_boot):
        pick = rng.choice(players, size=len(players), replace=True)
        idx = np.concatenate([idx_by_player[p] for p in pick])
        sub = df.loc[idx]
        pp = pure_premium(sub[freq_col].values, sub[matches_col].values, sev["mean_severity"])
        means.append(pp.mean())
    means = np.array(means)
    return {
        "mean": float(means.mean()),
        "lo95": float(np.percentile(means, 2.5)),
        "hi95": float(np.percentile(means, 97.5)),
        "sd": float(means.std(ddof=1)),
    }


# --------------------------------------------------------------------------
# Rating table
# --------------------------------------------------------------------------
FULL_CREDIBILITY_CLAIMS = 1082  # standard for +/-5% at 90% confidence

AGE_BANDS = [(0, 22), (22, 26), (26, 30), (30, 34), (34, 99)]
AGE_LABELS = ["<22", "22-25", "26-29", "30-33", "34+"]


def add_rating_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_band"] = pd.cut(
        out["age"],
        bins=[b[0] for b in AGE_BANDS] + [AGE_BANDS[-1][1]],
        labels=AGE_LABELS,
        right=False,
    )
    out["load_band"] = pd.qcut(
        out["matches_90d"], q=3, labels=["light", "moderate", "heavy"], duplicates="drop"
    )
    out["experience_band"] = pd.cut(
        out["career_matches"],
        bins=[-1, 100, 300, 1e9],
        labels=["<100 matches", "100-300", "300+"],
    )
    return out


def rating_table(
    df: pd.DataFrame,
    freq_col: str,
    matches_per_season: float,
    sev: dict,
    by: list[str],
    expense_loading: float = C.EXPENSE_LOADING,
) -> pd.DataFrame:
    """Premium table aggregated over rating cells, with observed experience."""
    g = df.groupby(by, observed=True).agg(
        exposure_matches=(freq_col, "size"),
        observed_events=("injury_event", "sum"),
        modelled_freq=(freq_col, "mean"),
    )
    g["observed_freq"] = g["observed_events"] / g["exposure_matches"]
    g["pure_premium"] = pure_premium(
        g["modelled_freq"].values, matches_per_season, sev["mean_severity"]
    )
    g["office_premium"] = office_premium(g["pure_premium"].values, expense_loading)
    g["ab_ratio"] = g["observed_freq"] / g["modelled_freq"]

    # Limited-fluctuation credibility. Full credibility for a frequency
    # estimate within +/-5% at 90% confidence requires ~1,082 claims; partial
    # credibility is the square root of the ratio. Almost every cell here has
    # two orders of magnitude fewer claims than that, which is precisely why
    # the premium is taken from the model rather than from the cell's own
    # observed rate.
    g["credibility"] = np.sqrt(g["observed_events"] / FULL_CREDIBILITY_CLAIMS).clip(
        upper=1.0
    )
    return g.reset_index()


# --------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------
def sensitivity_table(
    base_freq: float,
    matches_per_season: float,
    sev: dict,
    expense_loading: float = C.EXPENSE_LOADING,
) -> pd.DataFrame:
    """
    Sensitivity of the office premium to each assumption, one at a time.

    Deliberately NOT limited to "frequency +/- 10%". Because the premium is
    linear in frequency and in severity, a 10% shift in either moves the
    premium by exactly 10% - that calculation carries no information. What is
    worth testing is the assumptions where the relationship is non-obvious or
    where the plausible range is wide: the major/minor cost ratio, the exposure
    assumption (matches per season), and the loading basis.
    """
    base_pp = base_freq * matches_per_season * sev["mean_severity"]
    base_op = base_pp * (1 + expense_loading)

    rows = []

    def add(label, op):
        rows.append(
            {
                "scenario": label,
                "office_premium": op,
                "change_vs_base": op / base_op - 1,
            }
        )

    add("Base case", base_op)

    for d in (-0.20, -0.10, 0.10, 0.20):
        add(
            f"Event frequency {d:+.0%}",
            base_freq * (1 + d) * matches_per_season * sev["mean_severity"] * (1 + expense_loading),
        )

    for cost in (1_000, 1_500, 3_000, 5_000):
        add(
            f"Mean claim cost = £{cost:,}",
            base_freq * matches_per_season * cost * (1 + expense_loading),
        )

    for m in (20, 30, 60, 80):
        add(
            f"Matches per season = {m}",
            base_freq * m * sev["mean_severity"] * (1 + expense_loading),
        )

    for load in (0.10, 0.20, 0.30):
        add(f"Expense loading = {load:.0%}", base_pp * (1 + load))

    # The major:minor cost ratio changes the severity variance, not its mean,
    # so it leaves the pure premium untouched and only bites once a
    # variance-based risk loading is applied. Shown to make that explicit.
    for ratio in (2.0, 5.0, 10.0):
        p_major = sev["p_major"]
        c_minor = sev["mean_severity"] / (p_major * ratio + (1 - p_major))
        c_major = ratio * c_minor
        ex2 = p_major * c_major**2 + (1 - p_major) * c_minor**2
        en = base_freq * matches_per_season
        sd = np.sqrt(en * ex2)
        add(f"Major:minor cost ratio = {ratio:g} (with 0.1 SD risk load)", base_op + 0.1 * sd)

    return pd.DataFrame(rows)
