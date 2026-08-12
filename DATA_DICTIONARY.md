# Data Dictionary

Fields in `data/processed/player_match_panel.csv.gz`. One row = one player contesting one match = one unit of exposure.

`type` values: **raw** taken directly from the source match file; **derived** constructed in `src/features.py`; **TARGET** the modelled outcome.

All derived fields obey the no-look-ahead rule (METHODOLOGY 4.1): they are computable strictly before the match begins.

| field               | type             | description                                                                            | derived_from             |
|:--------------------|:-----------------|:---------------------------------------------------------------------------------------|:-------------------------|
| player_id           | raw              | Tour player identifier                                                                 | Source file              |
| season              | derived          | Calendar year of the tournament                                                        | tourney_date             |
| tourney_date        | raw              | Tournament start date                                                                  | Source file              |
| injury_event        | TARGET           | 1 if the player failed to complete or take the court (RET or W/O)                      | Parsed from score string |
| layoff_event        | secondary target | 1 if the player's next match is 90-540 days later (validation only, not a model input) | Gap to next match        |
| age                 | raw              | Player age in years at the tournament                                                  | Source file              |
| age_c_sq            | derived          | (age - 26)^2; centred to avoid VIF>100 against age                                     | age                      |
| height              | raw              | Height in cm; median-imputed within tour where missing                                 | Source file              |
| log_rank            | derived          | log(1 + singles ranking); unranked filled at rank 500                                  | Source file              |
| log_rank_points     | derived          | log(1 + ranking points)                                                                | Source file              |
| matches_28d         | derived          | Matches contested in the trailing 28 days, excluding this one                          | Rolling calendar window  |
| matches_90d         | derived          | Matches contested in the trailing 90 days                                              | Rolling calendar window  |
| matches_365d        | derived          | Matches contested in the trailing 365 days                                             | Rolling calendar window  |
| minutes_28d         | derived          | Court minutes in the trailing 28 days (see METHODOLOGY 3.4 on imputation)              | Rolling calendar window  |
| acwr                | derived          | Acute:chronic workload ratio - 28d daily load / 90d daily load, capped at 4            | minutes_28d, minutes_90d |
| days_since_last     | derived          | Days since the player's previous match, capped at 365                                  | tourney_date             |
| prev_match_minutes  | derived          | Duration of the player's previous match                                                | minutes                  |
| career_matches      | derived          | Cumulative tour matches before this one (experience proxy)                             | Cumulative count         |
| years_on_tour       | derived          | Years since the player's first match in the observation window                         | tourney_date             |
| prior_events_365d   | derived          | Non-completions in the trailing 365 days (injury history)                              | injury_event             |
| prior_events_career | derived          | Non-completions in career to date, excluding this match                                | injury_event             |
| round_ord           | derived          | Round as an ordinal, 0 (round robin) to 7 (final)                                      | round                    |
| match_in_event      | derived          | Match number within the current tournament (within-event fatigue)                      | Cumulative count         |
| surface_switch      | derived          | 1 if this match's surface differs from the player's previous match                     | surface                  |
| is_qualifier        | derived          | 1 if the player entered through qualifying                                             | entry                    |
| is_wildcard         | derived          | 1 if the player entered on a wildcard                                                  | entry                    |
| tour                | raw              | ATP or WTA                                                                             | Source file              |
| surface             | raw              | Hard / Clay / Grass; Carpet folded into Hard, casing normalised                        | Source file              |
| level               | derived          | grand_slam / masters / tour / finals / team / olympics; harmonised across tours        | tourney_level            |
| hand                | raw              | Playing hand: R / L / U (unknown)                                                      | Source file              |

## Source fields not carried forward

| Field | Why excluded |
|---|---|
| `minutes` (current match) | Generated by the match being predicted - leakage |
| `w_ace`, `l_df`, and all serve statistics | Same - post-match |
| `best_of` | r = 0.60 with the Grand Slam indicator; tier and tour dummies carry it |
| `minutes_90d` | VIF 39 against `matches_90d`, and depends on the minutes imputation |
| `load_per_day_28` | Exact rescaling of `minutes_28d` (VIF = infinity) |
| `draw_size` | Near-collinear with tournament tier |