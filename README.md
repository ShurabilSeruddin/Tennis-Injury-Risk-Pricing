# Tennis Injury Risk Modelling and Micro-Insurance Pricing

A frequency–severity pricing model for in-match injury risk on the ATP and WTA
tours, built end-to-end from public match records: data sourcing, feature
engineering, logistic regression, calibration testing, and an actuarial rating
structure with credibility and uncertainty analysis.

**163,918 player-match observations · 1,671 players · 2010–2025 · both tours**

---

## 1. Problem statement

An insurer is asked to write a one-season indemnity policy for a professional
tennis player, paying a fixed benefit each time the player fails to complete or
take the court for a scheduled tour match. To price it, three questions have to
be answered in order:

1. **How often does the event happen, and to whom?** Injury risk is not uniform.
   It should vary with age, accumulated workload, recovery time, and prior
   injury history — this is the standard model in the sports-medicine
   literature, and the question is whether it survives contact with data.
2. **What does the exposure measure look like?** A player contesting 60 matches
   a year carries roughly three times the annual risk of one contesting 20 at
   the same per-match rate. Get this wrong and the rating structure is wrong by
   a multiple, regardless of how good the model is.
3. **How confident can we be in the number?** A premium quoted without an
   interval is a point estimate presented as a fact.

This project answers all three from publicly available data, and is explicit
about where the answers are weak.

## 2. Headline results

| Metric | Value | Reading |
|---|---|---|
| Event rate (base) | 1.84% per player-match | ~1 non-completion per 54 matches |
| AUC-ROC (temporal holdout) | 0.651 | Modest ranking power |
| PR-AUC | 0.040 | 2.2× the base rate |
| **Decile lift** | **6.0×** | Riskiest tenth: 4.77% vs safest tenth: 0.80% |
| Brier score | 0.0182 | — |
| **Calibration (observed / predicted)** | **1.009** | Predicted probabilities usable as prices |
| 5-fold grouped CV AUC | 0.628 ± 0.015 | No material overfitting |
| Mean annual office premium | £1,334 | £338 (10th pct) to £2,291 (90th pct) |
| Portfolio premium, 95% bootstrap CI | £1,041 – £1,313 | Player-level resampling |

**The single most important line in that table is the calibration ratio, not
the AUC.** A pricing model is not a classifier. It never has to say yes or no;
it has to produce a number that can be multiplied by a claim cost. A model that
ranked risk brilliantly but ran 30% high on average would be useless for
pricing and would still show a strong AUC. This one predicts 1.84% and observes
1.86% out of sample, and holds within roughly ±10% across all ten deciles.

**Accuracy is deliberately not in that table.** At a 1.84% event rate, a model
that predicts "no injury" for every single match scores 98.16% accuracy. The
fitted model scores 98.14%. Reporting accuracy here would be actively
misleading, and the pipeline computes both numbers side by side to make that
concrete.

## 3. What the model found

Directionally consistent with the sports-medicine literature:

- **Age is the strongest continuous risk factor.** Players aged 34+ carry
  roughly twice the per-match risk of players under 26.
- **Prior injury history is the strongest individual predictor** (odds ratio
  1.35 per standard deviation of career prior events). Players who have failed
  to complete matches before do so again.
- **Match congestion raises risk**, but weakly, and the acute:chronic workload
  ratio adds less than the sports-science literature would suggest.
- **Tournament tier matters more than any player attribute.** Grand Slams and
  team events show materially lower non-completion rates than lower-tier
  events — an effect discussed in the limitations, since it is at least partly
  about incentives to finish rather than about physiology.

**One result runs the wrong way and is worth stating plainly:** matches played
in the trailing 365 days carries a *negative* coefficient — more tennis, less
risk. This is almost certainly reverse causation. Players who are fit play more
matches; players who are injured play fewer. The model is a rating tool, not a
causal one, and nothing in it supports the claim that playing more reduces
injury risk.

## 4. Repository structure

```
├── src/
│   ├── config.py          All assumptions and paths in one place
│   ├── ingest.py          Load, clean, harmonise ATP + WTA; explode to player-match
│   ├── features.py        Pre-match rolling workload / recovery / history features
│   ├── model.py           Logistic regression, GBM challenger, validation, VIF
│   ├── pricing.py         Frequency-severity, loadings, credibility, uncertainty
│   └── run_pipeline.py    End-to-end: raw files -> every table and figure
├── notebooks/
│   └── 01_analysis.ipynb  EDA and results walkthrough
├── docs/
│   ├── METHODOLOGY.md     Full technical write-up
│   └── DATA_DICTIONARY.md Every field, its source and its derivation
├── outputs/
│   ├── tables/            Premium tables, CV results, coefficients, sensitivity
│   └── figures/           EDA, calibration/ROC, feature importance, tornado
├── app/streamlit_app.py   Interactive premium estimator
└── data/processed/        Modelling panel (regenerated by the pipeline)
```

## 5. Reproducing

```bash
pip install -r requirements.txt
python -m src.run_pipeline          # ~3 minutes; clones source data on first run
streamlit run app/streamlit_app.py  # optional
```

Every number in this README and in `docs/METHODOLOGY.md` is written by
`run_pipeline.py` into `outputs/tables/results_summary.json`. None is typed by
hand.

## 6. Honest limitations

The full treatment is in `docs/METHODOLOGY.md` §7. The three that would stop
this being sold:

1. **The target is a proxy, not a diagnosis.** No public dataset records tennis
   injuries. What is recorded is that a player retired mid-match or withdrew.
   That misses every injury a player competes through, every injury sustained
   in practice or the off-season, and it wrongly includes tactical retirements
   by players already losing. §7.1 quantifies what can be quantified.
2. **The severity side is an assumption, not a finding.** The £2,000 mean claim
   cost has no empirical basis in this data — no public source prices tennis
   injury claims. Frequency is modelled; severity is assumed and stress-tested.
3. **This is professional tour data and it does not transfer to amateurs.**
   Recreational players differ in conditioning, medical access, playing
   schedule and match format. Fitting on professionals and pricing amateurs
   would be an extrapolation with no support in the data.

## 7. Data source and licence

Match records originate from the Tennis Abstract / Jeff Sackmann ATP and WTA
databases, under Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
**The original repositories are no longer publicly available**; this project
pulls from a community archive preserving the identical schema. The provenance
audit is in `docs/METHODOLOGY.md` §2. Non-commercial use only.
