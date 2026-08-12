# Methodology: Tennis Injury Risk Modelling and Micro-Insurance Pricing

Technical write-up covering data sourcing, processing decisions, model
construction, validation and pricing. Written to be read alongside the code in
`src/`; every figure quoted here is produced by `src/run_pipeline.py` and
persisted to `outputs/tables/results_summary.json`.

---

## 1. Motivation and product definition

The product is a one-season indemnity policy for a professional tennis player,
paying a fixed benefit each time the player fails to complete or take the court
for a scheduled tour match.

This is a deliberately narrow product, and the narrowness is the point. Broad
"injury insurance" cannot be priced from public data because injuries are not
publicly recorded. A non-completion, by contrast, is recorded in the match
result of every professional match played since 1968. Choosing a target that
is actually observable — and then being explicit about the gap between that
target and the thing you care about — is the central design decision in this
project.

The actuarial structure is standard:

```
Annual pure premium  =  E[N] × E[X]
                     =  (matches contested) × P(event | match) × E[cost per event]
Office premium       =  pure premium × (1 + expense loading)  [+ risk loading]
```

Three quantities therefore need estimating: the per-match event probability
(§4–5), the exposure measure (§6.1), and the severity distribution (§6.2).

---

## 2. Data sourcing, and a provenance problem

### 2.1 What the original project brief assumed

The brief specified: *"Find 2–3 tennis injury datasets (Kaggle, PubMed, sports
science databases), merge into a single clean CSV, minimum 200+ player
records"*, with features including *playing hours per week* and *years
experience*, and a target of *injury incidence*.

**This is not achievable as written, and the search for those datasets is what
produced the project's actual design.** The position after searching:

- Sports-medicine studies of tennis injury (Pluim et al.; the ATP/WTA medical
  reports) publish *aggregate incidence rates* — injuries per 1,000 match
  exposures, broken down by body region. They do not publish player-level
  records, because those are confidential medical data. An aggregate rate
  cannot be used to fit an individual-level model; there is nothing to fit to.
- Kaggle datasets tagged "tennis injury" are, on inspection, either
  match-results data with no injury field, or small hand-compiled tables of
  notable injuries to famous players — a sample selected on being newsworthy,
  which is the worst possible sampling frame for a frequency model.
- *Playing hours per week* is not recorded for professional players by any
  public source. Training load is private to each player's team.

Merging those sources into "a single clean CSV of 200+ player records" would
have produced a small, biased, largely fabricated dataset, and any model fitted
to it would have been a presentation exercise rather than an analysis.

### 2.2 The pivot

Instead, the project derives everything from **tour match records**, which are
public, complete, and large. This gives:

| Brief's intended feature | Public-data implementation |
|---|---|
| Age | Recorded directly per match |
| Playing hours/week | Court minutes in trailing 28 days; acute:chronic workload ratio |
| Years experience | Cumulative tour matches; years since first observed match |
| Surface type | Recorded directly; plus a surface-transition flag |
| Injury history | Prior non-completions in trailing 365 days and career to date |
| Injury incidence (target) | Non-completion of a scheduled match (RET / W/O) |

The result is 163,918 player-match observations across 1,671 players rather
than the 200 records the brief asked for — a difference of roughly three orders
of magnitude, which matters a great deal for a 1.84% event rate.

### 2.3 Provenance audit — the canonical source is offline

The standard public source for tour match data is Jeff Sackmann's
`tennis_atp` and `tennis_wta` GitHub repositories, used by essentially every
public tennis analytics project. **Both were unavailable at the time of
building this project.** Direct requests return 404; the author's GitHub
profile lists only `tennis_MatchChartingProject`. The repository README of the
surviving project notes prior licence violations and warns that updates may
stop, which is the plausible explanation.

This is worth recording for two reasons. First, reproducibility: any tutorial
or prior project pointing at those URLs is now broken. Second, and more
relevant to an insurer, it is a live illustration of **data provenance risk** —
a rating basis built on a single volunteer-maintained public source can vanish
without notice, and a pricing model whose data pipeline cannot be re-run is a
model that cannot be re-validated.

Data is therefore taken from a community archive
(`Aneeshers/tennis-sackmann-archive`) preserving the identical schema, with
`Tennismylife/TML-Database` cross-checked as an independent ATP mirror. Both
inherit the original CC BY-NC-SA 4.0 licence; use here is non-commercial.

### 2.4 Scope

Seasons 2010–2025 inclusive; 2026 excluded as a partial season. Both tours.
88,144 matches. Pre-2010 is excluded because court-time recording is
substantially sparser, and the workload features depend on it.

---

## 3. Processing pipeline

### 3.1 Harmonising the two tours

ATP and WTA files share a schema but not a vocabulary. Tournament tier is coded
`G/M/A/F/D/O` on the ATP side and `G/PM/P/I/W/F/O` on the WTA side. These were
mapped to a common six-level scale (`grand_slam`, `masters`, `tour`, `finals`,
`team`, `olympics`) so that one model can span both tours and a tour dummy can
then test for a residual difference. The mapping is a judgement call at the
Premier / Premier Mandatory boundary and is documented in `src/ingest.py`.

Surface arrives with inconsistent casing (`clay` and `Clay` both appear) and
with a near-extinct `Carpet` category (0.2% of rows, too few events to support
a coefficient), which is folded into `Hard`.

### 3.2 Deriving the target

The `score` field is free text. Non-completion is detected by regular
expression: `RET` for a mid-match retirement, `W/O` / `WALKOVER` / `DEF` for a
withdrawal before play. A small number of rows have blank or `UNFINISHED`
scores; these are dropped rather than coded as non-events, because coding
missing data as "no injury" would bias the frequency downward.

In this data the non-completing player is always recorded as the loser. That is
a definitional artefact of how results are recorded, not a finding — but it is
what makes the event attributable to a specific player, which the model needs.

Resulting rates, per match: RET 2.96% (ATP) / 2.88% (WTA), W/O 0.63% / 0.65%.

### 3.3 Exploding to player-match rows

Each match becomes two rows, one per participant. This is the modelling unit
and one unit of exposure. Both players are exposed to the risk of
non-completion in every match; only one can realise it, so the per-player-match
event rate is **1.84%** — half the per-match rate. ATP 1.83%, WTA 1.85%: the
two tours are almost indistinguishable on raw frequency, which is itself a
finding worth stating, since a naive prior might expect a difference from the
best-of-five format.

Walkovers are set to zero court minutes, since no tennis was played and the
match should not contribute to the player's subsequent workload history.

### 3.4 Missing court time — the largest cleaning decision

The `minutes` field is missing for 10.5% of ATP rows and **42.0% of WTA rows**.
This asymmetry is the single biggest data-quality problem in the project and it
is not missing at random: coverage is worse at smaller events, so dropping
those rows would bias the sample toward higher-tier tournaments and toward the
better-ranked players who play them.

Rows were therefore retained and minutes imputed from the median match duration
conditional on **(tour × best-of format × sets played)**, sets played being
parsed from the score string. Implausible values (≤10 minutes, >400 minutes,
which are data-entry errors rather than real matches) were treated as missing
and imputed the same way. An `minutes_imputed` flag is carried through so the
imputation is visible downstream.

The consequence for the model is handled in §4.2: minutes-based features are
deliberately given secondary weight relative to match-count features, which do
not depend on the imputation at all.

### 3.5 Burn-in

Rolling workload features are structurally zero for a player's first appearance
— not because they are rested, but because no history has been observed. Rows
are required to have at least five prior tour matches, removing 12,370
observations (7.0%). Without this the model would learn that "no recorded
workload" predicts low risk, which is an artefact of the observation window
rather than a property of the player.

---

## 4. Feature construction

### 4.1 The no-look-ahead rule

Every feature must be computable strictly *before* the match starts. The target
is whether the player completes *this* match, so any information generated by
the match itself — its duration, its result, what happened next — is leakage.

Mechanically, all rolling windows are computed within each player's history as
a calendar-time rolling sum with the current row subtracted out. Calendar
windows rather than match-count windows matter: five matches in ten days and
five matches in ten weeks are very different physical loads, and a
`rolling(5)` window cannot tell them apart.

Features constructed:

- **Workload:** matches and court minutes in trailing 28 / 90 / 365 days
- **Acute:chronic workload ratio** — 28-day daily load ÷ 90-day daily load. A
  standard sports-science construct: it measures load *relative to what the
  athlete is conditioned for* rather than load in absolute terms
- **Recovery:** days since last match; duration of the previous match
- **Experience:** cumulative tour matches; years since first observed match
- **Injury history:** prior non-completions in trailing 365 days and career
- **Context:** surface, surface switch since last match, tier, round,
  match number within the current tournament, qualifier/wildcard entry
- **Player:** age, centred age², height, handedness, log rank, log rank points

### 4.2 Collinearity screening — and what it changed

Variance inflation factors were computed on the full design matrix *before*
finalising the feature set. This is not a formality: the fitted coefficients
are a deliverable here, and a coefficient that is not separately identified
cannot be shown to a pricing committee.

The first pass was bad:

| Feature | VIF (first pass) | Action |
|---|---|---|
| `load_per_day_28` | ∞ | **Dropped** — an exact rescaling of `minutes_28d` (÷28) |
| `minutes_28d` | ∞ | Retained |
| `age_sq` | 110.3 | **Replaced** by age centred at 26 before squaring (r with age falls from 0.995 to near zero) |
| `age` | 108.7 | Retained |
| `minutes_90d` | 38.6 | **Dropped** — VIF 39 against `matches_90d`, and it depends on the imputation of §3.4, which `matches_90d` does not |
| `best_of` | — | **Dropped** — r = 0.60 with the Grand Slam indicator, since ATP Slams are the only best-of-five events; the tier and tour dummies already carry it |

The effect on the fitted model was material. In the first pass the Grand Slam
coefficient was −1.25 while `best_of` was +0.38, the two variables fighting
over the same signal and neither interpretable. After screening, the tier
coefficients are stable and orderable.

**Residual issue, disclosed:** `matches_28d` (VIF 20.3) and `minutes_28d`
(18.8) remain correlated. Both are retained because they measure genuinely
different things — how many times you competed versus how long you were on
court — but their *individual* coefficients should not be read separately. The
joint effect of recent workload is interpretable; the split between count and
duration is not.

---

## 5. Model development and validation

### 5.1 Validation protocol

**The brief specified a random 70/30 split. That is the wrong protocol here,
and the project uses a temporal holdout instead: fit on 2010–2022, test on
2023–2025.**

The reason is that rows are repeated observations on the same 1,671 players.
A random split puts a player's January match in training and their February
match in test, so a model can score well partly by memorising player-specific
risk rather than learning transferable structure. More fundamentally, the
operational task is forecasting: an insurer prices next season using past
seasons, so validation should reproduce that.

Cross-validation uses `GroupKFold` on `player_id` for the same reason — a
player appears in exactly one fold.

**The honest result:** the random split was also run, expecting it to look
optimistically better. It did not. Random-split test AUC was 0.629 against
0.651 for the temporal split — the naive protocol scored *worse*, by 0.022.
The features are apparently not player-identifying enough for random splitting
to leak much in this dataset. The temporal split remains the correct protocol
on principle, but the specific concern that motivated it did not bind here, and
reporting it the other way would have been reporting a prediction rather than a
result.

### 5.2 Models

- **Logistic regression (L2)** — the production model. Chosen for
  interpretability, because the coefficients are the rating factors.
- **Histogram gradient boosting** — a challenger, to measure what the linear
  form gives up.
- **Intercept-only baseline** — the honest floor.

Class weighting was *not* used. Balancing improves recall but destroys
calibration, and calibrated probabilities are the entire deliverable.

### 5.3 Results

| | Train | Test (2023–25) | GBM (test) | Random-split |
|---|---|---|---|---|
| AUC-ROC | 0.633 | **0.651** | 0.668 | 0.629 |
| PR-AUC | 0.032 | **0.040** | 0.040 | 0.031 |
| Brier | 0.0180 | **0.0182** | 0.0181 | 0.0180 |
| Decile lift | 5.4× | **6.0×** | 6.5× | 4.5× |
| Observed / predicted | 1.000 | **1.009** | 1.020 | 1.001 |

Five-fold grouped CV: AUC **0.628 ± 0.015**.

**Overfitting:** none detectable. Test AUC (0.651) exceeds train AUC (0.633),
and CV fold-to-fold standard deviation is 0.015. The model is, if anything,
underfitting — which is the expected outcome for a genuinely noisy target.

**On the AUC of 0.65.** This is modest and should not be dressed up. It means
that given one player who fails to complete and one who does not, the model
ranks them correctly 65% of the time. That is well above chance and consistent
with the sports-medicine literature, where individual injury prediction models
typically report AUCs in the 0.6–0.7 range: a large share of injury risk is
genuinely stochastic — a bad step, an awkward landing — and no amount of
feature engineering on public data will recover it.

**Why the model is nonetheless usable for pricing.** Two reasons.

First, **decile lift of 6.0×**. The riskiest tenth of player-matches
experiences non-completion at 4.77% against 0.80% for the safest tenth. Rating
does not require identifying *which* player gets hurt; it requires separating
groups whose average frequency differs. A 6× spread across rating classes is
commercially meaningful.

Second, **calibration**. Across all ten predicted-probability deciles, observed
frequency tracks predicted within roughly ±10%, with an overall ratio of 1.009.
This is the property that makes the output a price rather than a score.

**Why the logistic regression is preferred despite the GBM's higher AUC.** The
GBM wins on AUC (0.668 vs 0.651) but ties on PR-AUC (0.040) and calibrates
worse (1.020 vs 1.009). For a 0.017 AUC gain, it costs full coefficient
interpretability. In a rating context where every factor has to be defensible
to a regulator and explicable to an underwriter, that is not a trade worth
making. The GBM's value is diagnostic: its narrow margin says the linear form
is not leaving much on the table.

### 5.4 Why accuracy is excluded

At a 1.84% event rate, predicting "no injury" for every match yields **98.16%**
accuracy. The fitted model yields **98.14%**. Both numbers are computed by
`src/model.py` and reported side by side deliberately. Any evaluation framework
that rewards accuracy on this problem selects for a model that does nothing.

### 5.5 Does the target behave like injury?

The target is a proxy, so it needs a sanity check independent of the model. If
non-completions are genuinely injury-driven, they should be followed by
extended absences more often than completed matches are.

- P(90+ day absence | player did not complete) = **9.78%**
- P(90+ day absence | player completed) = **3.67%**
- **Lift = 2.66×**

Directionally right and statistically clear across 2,955 events. But a 2.66×
lift is not overwhelming, and 90% of non-completions are *not* followed by an
extended absence. That is consistent with most being minor — cramp, heat
illness, a strain a player recovers from within days — and with some being
tactical rather than physical. This is the empirical basis for the two-point
severity mix in §6.2.

---

## 6. Actuarial pricing

### 6.1 Exposure — the step the brief omitted

The brief specified `pure premium = P(injury) × cost`. This omits exposure, and
the omission is not cosmetic: without it, a player contesting 60 matches and
one contesting 20 are charged the same annual premium for three times the
annual risk.

Defining exposure required a genuine decision. The distribution of matches per
season is severely right-skewed:

- **Median matches per season, per player: 5.7**
- **Median matches per season, weighted by matches (i.e. for the player who
  contests a typical tour match): 30.8**

The first number is dominated by the long tail of qualifiers who appear once
and never return. They are in the data but they are not the population an
insurer would write. **The rating basis is therefore 30.8 matches per season**,
and individual premiums use each player's own observed average. Had the naive
median been used, every premium in this report would be understated by a factor
of five — which is a larger error than anything the model itself could make.

### 6.2 Severity

The brief specified a flat £2,000 per injury. Retaining that as the *portfolio
mean* is reasonable as a placeholder, but applying it flat would make the
premium table an exact rescaling of the frequency table, testing nothing.

Severity is therefore a two-point mix, split on the observed data from §5.5:

| Component | Probability | Assumed cost |
|---|---|---|
| Major (followed by 90+ day absence) | 9.8% *(observed)* | £7,188 |
| Minor | 90.2% *(observed)* | £1,438 |
| **Mean** | | **£2,000** *(assumed)* |

Coefficient of variation: 0.85.

The split probability is empirical. The **5:1 cost ratio between major and
minor is an assumption** — no public source prices tennis injury claims — and
is stress-tested at 2:1 and 10:1 in §6.5. Because the mix is calibrated to hold
the mean at £2,000, the ratio changes the *variance* of severity but not the
pure premium; it bites only once a variance-based risk loading is applied.

### 6.3 Premiums

With a 15% expense loading:

| | Value |
|---|---|
| Mean pure premium | £1,160 |
| Mean office premium | **£1,334** |
| 10th percentile | £338 |
| 90th percentile | £2,291 |

The premium range across the book is roughly 7:1, which comes from the
combination of the 6× frequency spread and the variation in matches contested.

### 6.4 Uncertainty

Two distinct sources, quantified separately.

**Parameter uncertainty** — how well the rate itself is known. sklearn does not
expose standard errors, so the asymptotic covariance of the logistic
coefficients was reconstructed from the Fisher information, `(X'WX)⁻¹`, and
propagated through the linear predictor:

> Mean pure premium: **£1,160** (95% CI **£927 – £1,456**)
> Median relative interval width: **42% of the central estimate**

**Portfolio uncertainty** — bootstrap resampling of *players*, not rows,
because a player's matches are correlated with one another and row-level
resampling would understate the interval:

> Portfolio mean premium: **£1,167** (95% CI **£1,041 – £1,313**), SD £69

**Process variance** — the volatility of one player's actual year — is separate
again and dominates both. Under a compound Poisson model with `Var(S) = E[N]·E[X²]`,
a typical player's aggregate loss has a standard deviation several times its
mean, which is exactly why the risk exists to be insured. A 0.1-standard-deviation
risk loading is provided as an option in `src/pricing.py`.

Note that Poisson is an approximation. Injury risk is contagious — an injured
player is more likely to be injured again. The `prior_events` features capture
part of this, but residual contagion would make the true distribution
over-dispersed relative to Poisson, meaning the process variance above is an
**understatement**.

### 6.5 Sensitivity analysis

The brief asked how the premium changes if injury probability moves ±10%. The
answer is arithmetic: the premium is linear in frequency, so it moves ±10%
exactly. The same is true of mean claim cost. Reporting those two figures
produces no information.

The analysis was therefore restructured to test the assumptions where the
plausible range is *wide* rather than where the relationship is unknown:

| Assumption | Range tested | Premium impact |
|---|---|---|
| Mean claim cost | £1,000 – £5,000 | **−50% to +150%** |
| Matches per season | 20 – 80 | **−35% to +159%** |
| Event frequency | ±20% | ±20% (linear, as expected) |
| Expense loading | 10% – 30% | −4% to +13% |
| Major:minor cost ratio | 2:1 – 10:1 | +12% to +20% *(via risk loading only)* |

**The conclusion is uncomfortable and worth stating.** The two assumptions that
dominate the premium — the mean claim cost and the number of matches a player
contests — are the two the model does not estimate. The claim cost is asserted;
the match count is a forecast of the player's own schedule. Model refinement
that improved AUC from 0.65 to 0.70 would move premiums by a few percent. Being
wrong about the claim cost by a factor of two moves them by a factor of two.
For this product, **the modelling is not the binding constraint on pricing
accuracy** — and identifying that is more useful than another decimal of AUC.

### 6.6 Credibility

Every cell of the rating table carries a limited-fluctuation credibility
factor. Full credibility for a frequency estimate within ±5% at 90% confidence
requires ~1,082 claims; partial credibility is `√(claims / 1,082)`.

Cells contain between 5 and 80 observed events, giving credibility factors of
**0.07 to 0.27**. No cell is remotely credible on its own experience. This is
the quantitative justification for taking premiums from the fitted model rather
than from cell-level observed rates — and it is why the actual/expected ratios
in the rating table range from 0.47 to 1.47 without that being evidence of
model failure. The 34+/heavy-load cell, with an A/E of 0.47, contains five
claims.

---

## 7. Limitations

### 7.1 The target is a proxy, not a diagnosis

This is the fundamental limitation and no amount of modelling fixes it.

**Under-counting.** The target misses every injury a player competes through,
every injury sustained in training or the off-season, and every withdrawal
before a draw is published — which is how many injuries actually present. The
true injury incidence is certainly higher than 1.84% per player-match.

**Over-counting.** Not all retirements are injuries. Retiring while losing
badly preserves the body for the next event, and this behaviour is documented
in tennis economics research. Some fraction of the target is tactical. It
cannot be separated out with public data.

**Directional bias.** Both errors likely correlate with the same features the
model uses. Higher-ranked players have more medical support and more incentive
to complete a Grand Slam match. The strong negative Grand Slam coefficient
(OR 0.46) is plausibly incentive rather than physiology: prize money and
ranking points are largest exactly where the model sees the fewest
non-completions. A pricing model that mistakes an incentive effect for a risk
effect will underprice Grand Slam exposure.

### 7.2 Reverse causation in the workload features

`matches_365d` carries a negative coefficient (OR 0.86): more tennis, less
risk. This is the healthy-worker effect. Fit players accumulate matches;
injured players do not. The model is predictive, not causal, and nothing in it
supports advice to play more.

This also caps what the workload features can tell us. The acute:chronic
workload ratio contributes less than the sports-science literature would
predict, and part of the reason is that the same confounding runs through it.

### 7.3 Severity is assumed

The entire severity side rests on an assumed £2,000 mean, and §6.5 shows it is
one of the two dominant drivers of the premium. Before quoting this product,
an insurer would need actual claims data. Nothing here substitutes for that.

### 7.4 This does not transfer to amateur players

The model is fitted on professional tour players. Recreational players differ
in conditioning, medical access, playing frequency, match format and — most
importantly — in whether anyone records their retirements at all. Applying
these rating factors to an amateur book would be extrapolation beyond the
support of the data.

### 7.5 Insurability

Setting the modelling aside, the product as specified has structural problems
an underwriter would raise immediately:

- **Adverse selection.** Players know their own physical condition far better
  than any public-data model does. A player nursing a strain has an obvious
  incentive to buy, and no feature in this model would detect it.
- **Moral hazard.** The benefit pays on non-completion, and the player controls
  whether they retire. Paying someone to stop playing is a design flaw
  independent of pricing accuracy.
- **Correlation.** Scheduling and surface transitions are common across the
  tour, so claims are not independent — a compressed clay swing raises risk for
  the whole book at once.

None of these is a modelling failure. All would need addressing through policy
design — waiting periods, medical underwriting, benefit caps — before the
rating structure mattered.

### 7.6 Other

- Structural change: the model is fitted across 2010–2025, a period spanning
  rule changes (final-set tiebreaks), the COVID-affected 2020 season (excluded
  from the source files) and shifts in tour scheduling. Stationarity is assumed
  and not tested.
- Doubles, ITF, Challenger and qualifying matches are excluded, so workload is
  understated for lower-ranked players who play more of them.
- Height is missing for some lower-ranked players and is median-imputed within
  tour.

---

## 8. What I would do next

In order of expected value, not of technical interest:

1. **Get real claims data.** §6.5 shows this dominates everything else.
2. **Model severity properly** as days-out, using the observed absence
   distribution, rather than as a two-point mix calibrated to an assumption.
3. **Survival framing.** Time-to-next-injury with right-censoring is the
   natural structure and would use the layoff data currently held back for
   validation.
4. **Player random effects.** A hierarchical GLM with a player-level random
   intercept would give an explicit credibility structure — Bühlmann–Straub in
   a form a pricing committee would recognise — instead of the pooled fit here.
5. **Separate RET from W/O.** They are different events with different lead
   times; pooling them was a simplification.

---

## 9. Reproducibility

```bash
pip install -r requirements.txt
python -m src.run_pipeline
```

Seeded at `RANDOM_STATE = 42`. All assumptions in `src/config.py`. All results
written to `outputs/tables/results_summary.json`; no number in this document is
entered by hand.
