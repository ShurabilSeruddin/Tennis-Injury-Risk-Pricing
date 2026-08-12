"""
End-to-end pipeline: raw match files -> premium tables.

Run with:  python -m src.run_pipeline

Everything written to outputs/ is reproducible from this script and the seed in
config.py. No number in the report is typed by hand.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import model as M
from . import pricing as P
from .features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_features,
    model_frame,
)
from .ingest import load_matches, to_player_matches

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

RESULTS: dict = {}


def step(msg):
    print(f"\n=== {msg} ===")


def main():
    # ---------------------------------------------------------------- 1
    step("1. Ingest")
    matches = load_matches()
    panel = to_player_matches(matches)
    print(f"matches: {len(matches):,}  player-match rows: {len(panel):,}")
    RESULTS["n_matches"] = len(matches)
    RESULTS["n_player_matches"] = len(panel)

    # ---------------------------------------------------------------- 2
    step("2. Features")
    feat = build_features(panel)
    df = model_frame(feat)
    print(f"modelling rows: {len(df):,}  players: {df['player_id'].nunique():,}")
    print(f"event rate: {df[TARGET].mean():.3%}")
    RESULTS["n_modelling_rows"] = len(df)
    RESULTS["n_players"] = int(df["player_id"].nunique())
    RESULTS["event_rate"] = float(df[TARGET].mean())
    RESULTS["event_rate_atp"] = float(df.loc[df.tour == "ATP", TARGET].mean())
    RESULTS["event_rate_wta"] = float(df.loc[df.tour == "WTA", TARGET].mean())

    df.to_csv(C.DATA_PROCESSED / "player_match_panel.csv.gz", index=False)

    # ---------------------------------------------------------------- 3
    step("3. EDA")
    eda_by_age = (
        df.assign(age_band=pd.cut(df["age"], [0, 22, 26, 30, 34, 99],
                                  labels=P.AGE_LABELS, right=False))
        .groupby("age_band", observed=True)
        .agg(n=(TARGET, "size"), events=(TARGET, "sum"), rate=(TARGET, "mean"))
    )
    eda_by_age.to_csv(C.OUT_TAB / "eda_event_rate_by_age.csv")
    print(eda_by_age)

    eda_by_surface = df.groupby(["tour", "surface"]).agg(
        n=(TARGET, "size"), rate=(TARGET, "mean")
    )
    eda_by_surface.to_csv(C.OUT_TAB / "eda_event_rate_by_surface.csv")

    # Workload gradient - the headline EDA chart.
    df["_load_decile"] = pd.qcut(df["matches_28d"], 10, labels=False, duplicates="drop")
    load_grad = df.groupby("_load_decile").agg(
        matches_28d=("matches_28d", "mean"), rate=(TARGET, "mean"), n=(TARGET, "size")
    )
    load_grad.to_csv(C.OUT_TAB / "eda_event_rate_by_workload.csv")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    eda_by_age["rate"].mul(100).plot(kind="bar", ax=ax[0], color="#3b6ea5")
    ax[0].set_title("Event rate by age band"); ax[0].set_ylabel("% of matches")
    ax[0].tick_params(axis="x", rotation=0)
    ax[1].plot(load_grad["matches_28d"], load_grad["rate"] * 100, "o-", color="#c0504d")
    ax[1].set_title("Event rate by 28-day match load")
    ax[1].set_xlabel("matches in prior 28 days"); ax[1].set_ylabel("% of matches")
    rest = df.assign(rb=pd.cut(df["days_since_last"], [0, 3, 7, 14, 30, 90, 400]))
    rr = rest.groupby("rb", observed=True)[TARGET].mean() * 100
    rr.plot(kind="bar", ax=ax[2], color="#7f9a3c")
    ax[2].set_title("Event rate by days since last match")
    ax[2].set_ylabel("% of matches"); ax[2].tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(C.OUT_FIG / "eda_overview.png"); plt.close(fig)

    step("3b. Collinearity screen")
    vif = M.vif_table(df)
    vif.to_csv(C.OUT_TAB / "vif_diagnostics.csv", index=False)
    print(vif.head(10).round(2).to_string(index=False))
    RESULTS["max_vif"] = float(vif["vif"].replace([np.inf], np.nan).max())

    # ---------------------------------------------------------------- 4
    step("4. Modelling - temporal holdout")
    train, test = M.temporal_split(df)
    print(f"train {len(train):,} rows ({train.season.min()}-{train.season.max()})  "
          f"test {len(test):,} rows ({test.season.min()}-{test.season.max()})")
    RESULTS["n_train"] = len(train); RESULTS["n_test"] = len(test)

    logit, m_tr, m_te, p_te = M.fit_and_score(M.make_logit(), train, test)
    print("logit  train:", {k: round(v, 4) for k, v in m_tr.items() if k in
                            ("auc_roc", "pr_auc", "brier")})
    print("logit  test :", {k: round(v, 4) for k, v in m_te.items() if k in
                            ("auc_roc", "pr_auc", "brier", "obs_over_pred")})
    print(f"       decile lift {m_te['decile_lift']:.2f}x  "
          f"(top {m_te['top_decile_rate']:.2%} vs bottom {m_te['bottom_decile_rate']:.2%})")
    RESULTS["logit_train"] = m_tr; RESULTS["logit_test"] = m_te

    gbm, g_tr, g_te, p_gbm = M.fit_and_score(M.make_gbm(), train, test)
    print("gbm    test :", {k: round(v, 4) for k, v in g_te.items() if k in
                            ("auc_roc", "pr_auc", "brier", "obs_over_pred")})
    RESULTS["gbm_train"] = g_tr; RESULTS["gbm_test"] = g_te

    # Intercept-only baseline: the honest floor any model must beat.
    base_p = np.full(len(test), train[TARGET].mean())
    RESULTS["baseline_test"] = {
        "brier": float(np.mean((test[TARGET].values - base_p) ** 2)),
        "accuracy_predict_none": float(1 - test[TARGET].mean()),
    }

    # ---------------------------------------------------------------- 5
    step("5. Leakage demonstration - naive random 70/30 split")
    from sklearn.model_selection import train_test_split

    rtr, rte = train_test_split(df, test_size=0.30, random_state=C.RANDOM_STATE,
                                stratify=df[TARGET])
    _, _, m_rand, _ = M.fit_and_score(M.make_logit(), rtr, rte)
    print(f"random-split test AUC   {m_rand['auc_roc']:.4f}")
    print(f"temporal-split test AUC {m_te['auc_roc']:.4f}")
    print("NOTE: report whichever way this falls. The temporal split is the")
    print("correct protocol on principle regardless of whether the random")
    print("split turns out to be optimistic in this particular dataset.")
    RESULTS["random_split_test"] = m_rand
    RESULTS["leakage_auc_gap"] = float(m_rand["auc_roc"] - m_te["auc_roc"])

    # ---------------------------------------------------------------- 6
    step("6. Cross-validation (GroupKFold on player)")
    cv = M.grouped_cv(M.make_logit, df)
    cv.to_csv(C.OUT_TAB / "cv_results.csv", index=False)
    print(cv[["fold", "auc_roc", "pr_auc", "brier"]].round(4).to_string(index=False))
    print("mean AUC %.4f  sd %.4f" % (cv.auc_roc.mean(), cv.auc_roc.std(ddof=1)))
    RESULTS["cv_auc_mean"] = float(cv.auc_roc.mean())
    RESULTS["cv_auc_sd"] = float(cv.auc_roc.std(ddof=1))
    RESULTS["cv_pr_auc_mean"] = float(cv.pr_auc.mean())

    # ---------------------------------------------------------------- 7
    step("7. Coefficients / feature importance")
    coefs = M.coefficient_table(logit)
    coefs.to_csv(C.OUT_TAB / "logit_coefficients.csv", index=False)
    print(coefs.head(15).round(4).to_string(index=False))

    top = coefs.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    cols = ["#c0504d" if c > 0 else "#3b6ea5" for c in top["coefficient"]]
    ax.barh(top["feature"], top["coefficient"], color=cols)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("logistic coefficient (per 1 SD; categoricals vs reference)")
    ax.set_title("Drivers of non-completion risk")
    fig.tight_layout(); fig.savefig(C.OUT_FIG / "feature_importance.png"); plt.close(fig)

    # ---------------------------------------------------------------- 8
    step("8. Calibration")
    cal = M.calibration_table(test[TARGET].values, p_te)
    cal.to_csv(C.OUT_TAB / "calibration_deciles.csv", index=False)
    print(cal.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(cal["predicted"] * 100, cal["observed"] * 100, "o-", color="#3b6ea5",
               label="logistic")
    lim = max(cal["predicted"].max(), cal["observed"].max()) * 100 * 1.1
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.8, label="perfect")
    ax[0].set_xlabel("predicted %"); ax[0].set_ylabel("observed %")
    ax[0].set_title("Calibration, temporal holdout"); ax[0].legend()

    from sklearn.metrics import roc_curve
    for nm, pp, cc in [("Logistic", p_te, "#3b6ea5"), ("Gradient boosting", p_gbm, "#c0504d")]:
        fpr, tpr, _ = roc_curve(test[TARGET].values, pp)
        ax[1].plot(fpr, tpr, color=cc, label=f"{nm}")
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("true positive rate")
    ax[1].set_title("ROC, temporal holdout"); ax[1].legend()
    fig.tight_layout(); fig.savefig(C.OUT_FIG / "calibration_and_roc.png"); plt.close(fig)

    # ---------------------------------------------------------------- 9
    step("9. Severity model")
    sev = P.fit_severity(df)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sev.items()})
    RESULTS["severity"] = sev

    # --------------------------------------------------------------- 10
    step("10. Exposure: matches per season")
    mps = (
        df.groupby(["player_id", "season"]).size().rename("m").reset_index()
        .groupby("player_id")["m"].mean()
    )
    # Two very different numbers, and the choice between them changes every
    # premium in the report. The per-PLAYER median is dragged down by the long
    # tail of qualifiers who appear once; the MATCH-WEIGHTED median describes
    # the player who contests a typical tour match, which is the population an
    # insurer would actually be writing.
    mw = df[["player_id"]].merge(mps.rename("m"), on="player_id", how="left")["m"]
    RESULTS["matches_per_season_player_median"] = float(mps.median())
    RESULTS["matches_per_season_mean"] = float(mps.mean())
    RESULTS["matches_per_season_median"] = float(mw.median())
    print(f"per-player median {mps.median():.1f} | match-weighted median "
          f"{mw.median():.1f} <- rating basis")

    test = test.copy()
    test["freq"] = p_te
    test = test.merge(mps.rename("matches_per_season"), on="player_id", how="left")
    test["matches_per_season"] = test["matches_per_season"].fillna(mps.median())

    # --------------------------------------------------------------- 11
    step("11. Premiums")
    test["pure_premium"] = P.pure_premium(
        test["freq"], test["matches_per_season"], sev["mean_severity"]
    )
    test["agg_sd"] = P.aggregate_sd(test["freq"], test["matches_per_season"], sev)
    test["office_premium"] = P.office_premium(test["pure_premium"], C.EXPENSE_LOADING)
    test["office_premium_riskloaded"] = P.office_premium(
        test["pure_premium"], C.EXPENSE_LOADING, risk_loading_sd=0.10, sd=test["agg_sd"]
    )
    print(test[["freq", "matches_per_season", "pure_premium", "office_premium",
                "office_premium_riskloaded"]].describe().round(2).to_string())
    RESULTS["mean_pure_premium"] = float(test["pure_premium"].mean())
    RESULTS["mean_office_premium"] = float(test["office_premium"].mean())
    RESULTS["premium_p10"] = float(test["office_premium"].quantile(0.10))
    RESULTS["premium_p90"] = float(test["office_premium"].quantile(0.90))

    # Parameter-uncertainty interval via Fisher information
    prep = logit.named_steps["prep"]
    Xtr = prep.transform(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    Xte = prep.transform(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    Xtr = np.hstack([np.ones((Xtr.shape[0], 1)), np.asarray(Xtr)])
    Xte = np.hstack([np.ones((Xte.shape[0], 1)), np.asarray(Xte)])
    p_tr_hat = logit.predict_proba(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    cov = P.logit_parameter_covariance(logit, Xtr, p_tr_hat)
    eta = np.log(test["freq"] / (1 - test["freq"])).values
    se_eta = np.sqrt(np.einsum("ij,jk,ik->i", Xte, cov, Xte))
    ci = P.premium_interval_analytic(eta, se_eta, test["matches_per_season"].values,
                                     sev["mean_severity"])
    RESULTS["mean_pp_ci"] = [float(ci.pp_lo.mean()), float(ci.pp_mid.mean()),
                             float(ci.pp_hi.mean())]
    RESULTS["median_relative_ci_width"] = float(
        ((ci.pp_hi - ci.pp_lo) / ci.pp_mid).median()
    )
    print(f"mean pure premium 95% CI (parameter uncertainty): "
          f"£{ci.pp_lo.mean():.0f} - £{ci.pp_hi.mean():.0f}")

    boot = P.bootstrap_portfolio(test.reset_index(drop=True), "freq",
                                 "matches_per_season", sev)
    RESULTS["portfolio_bootstrap"] = boot
    print("portfolio bootstrap:", {k: round(v, 1) for k, v in boot.items()})

    # --------------------------------------------------------------- 12
    step("12. Rating tables")
    banded = P.add_rating_bands(test)
    t1 = P.rating_table(banded, "freq", RESULTS["matches_per_season_median"], sev,
                        ["age_band", "load_band"])
    t1.to_csv(C.OUT_TAB / "premium_table_age_x_load.csv", index=False)
    print(t1.round(4).to_string(index=False))

    t2 = P.rating_table(banded, "freq", RESULTS["matches_per_season_median"], sev,
                        ["tour", "age_band"])
    t2.to_csv(C.OUT_TAB / "premium_table_tour_x_age.csv", index=False)

    t3 = P.rating_table(banded, "freq", RESULTS["matches_per_season_median"], sev,
                        ["experience_band", "age_band"])
    t3.to_csv(C.OUT_TAB / "premium_table_experience_x_age.csv", index=False)

    piv = t1.pivot(index="age_band", columns="load_band", values="office_premium")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"£{v:,.0f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Annual office premium (£{sev['mean_severity']:,.0f} mean claim,\n"
                 f"{RESULTS['matches_per_season_median']:.0f} matches, "
                 f"{C.EXPENSE_LOADING:.0%} loading)")
    ax.set_xlabel("90-day match load"); ax.set_ylabel("age band"); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(C.OUT_FIG / "premium_heatmap.png"); plt.close(fig)

    # --------------------------------------------------------------- 13
    step("13. Sensitivity")
    sens = P.sensitivity_table(float(test["freq"].mean()),
                               RESULTS["matches_per_season_median"], sev)
    sens.to_csv(C.OUT_TAB / "sensitivity_analysis.csv", index=False)
    print(sens.round(4).to_string(index=False))

    tor = sens[sens.scenario != "Base case"].copy()
    tor = tor.reindex(tor.change_vs_base.abs().sort_values().index).tail(14)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.barh(tor["scenario"], tor["change_vs_base"] * 100,
            color=["#c0504d" if v > 0 else "#3b6ea5" for v in tor["change_vs_base"]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("% change in office premium vs base case")
    ax.set_title("Premium sensitivity to assumptions")
    fig.tight_layout(); fig.savefig(C.OUT_FIG / "sensitivity_tornado.png"); plt.close(fig)

    # --------------------------------------------------------------- 14
    step("14. Secondary target check (extended absence)")
    sec = df[df["layoff_event"].notna()].copy()
    lay_rate_event = sec.loc[sec[TARGET] == 1, "layoff_event"].mean()
    lay_rate_noevent = sec.loc[sec[TARGET] == 0, "layoff_event"].mean()
    print(f"P(90d+ absence | non-completion) = {lay_rate_event:.3%}")
    print(f"P(90d+ absence | completed)      = {lay_rate_noevent:.3%}")
    RESULTS["layoff_given_event"] = float(lay_rate_event)
    RESULTS["layoff_given_no_event"] = float(lay_rate_noevent)
    RESULTS["layoff_lift"] = float(lay_rate_event / lay_rate_noevent)

    # --------------------------------------------------------------- 15
    step("15. Persist")
    import joblib
    joblib.dump(logit, C.MODELS / "logit_pipeline.joblib")
    joblib.dump({"severity": sev,
                 "matches_per_season_median": RESULTS["matches_per_season_median"]},
                C.MODELS / "pricing_params.joblib")
    test.to_csv(C.OUT_TAB / "scored_test_set.csv.gz", index=False)
    with open(C.OUT_TAB / "results_summary.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print(f"written to {C.OUT_TAB} and {C.OUT_FIG}")


if __name__ == "__main__":
    main()
