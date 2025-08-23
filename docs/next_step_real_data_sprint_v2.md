# Next Step — Multi-Season Training, Label Hygiene, and Backtests

## Goal

Train on 2022–2024 regular seasons with clean IL labels, compute stable features, run rolling backtests, calibrate, and ship a richer /risk payload. Keep SQLite for dev; prep Postgres for scale.

## 0) Prereqs (quick)

- [x] Create branch feat/multiseason-training.
- [x] Snapshot DB: cp pitchguard_dev.db pitchguard_dev_pre_multiseason.db.
- [x] Add runbook skeleton ops/runbooks/multiseason_training.md.

## 1) Expand Real Data (2022–2024)

### What to do

Load pitches and appearances for 2022, 2023, 2024 regular seasons.

Ensure idempotent loads: using a (pitcher_id, game_date, at_bat_id, pitch_number) primary key (or unique index) to avoid dupes.

### Implementation notes

Batch by month to avoid giant pulls.

After each load, derive/update appearances (per pitcher per game aggregates).

### Success checks

Row counts by season (store in docs/data_ingest_report.md):

- pitches: ~1.9–2.3M/season (MLB scale; your dev subset can be smaller, just record it).
- appearances: ~10–12K/season.

Missingness:

- % NULL release_speed < 3%
- % NULL release_spin_rate < 5%

Indices (even on SQLite)

```sql
CREATE INDEX IF NOT EXISTS idx_pitches_pid_date ON pitches(pitcher_id, game_date);
CREATE INDEX IF NOT EXISTS idx_appearances_pid_date ON appearances(pitcher_id, game_date);
```

## 2) Injury / IL Label Hygiene

### Why: cleaner positives → higher ceiling.

### What to do

Build injuries with IL starts/ends from transactions.

Normalize text (lower, strip punctuation).

Maintain whitelists/blacklists in docs/injury_filters.md:

- Include (arm/shoulder/elbow/forearm/lat/rotator cuff/UCL; "strain", "inflammation", "soreness").
- Exclude (illness, flu, COVID, concussion, HBP fractures, hamstring, paternity, bereavement, disciplinary).

### Snapshot label policy

Per-appearance snapshots (recommended): for each (pitcher_id, game_date), label 1 if any IL start in (game_date, game_date + 21d] after that appearance; else 0.

### Success checks

Label prevalence report (per season, overall) in docs/label_report.md:

- Positive rate typically low (<< 5%); exact rate logged.
- Manual QA: random 25 positives → at least 90% are arm-related.

## 3) Feature Set (≤ 40; zero leakage)

### Families & examples

#### Acute & Chronic Workload

- last_game_pitches, last_3_games_pitches
- roll7d_pitch_count, roll14d_pitch_count, roll30d_pitch_count
- roll7d_appearances, roll14d_appearances

#### Recovery

- days_since_last_appearance, avg_rest_days_60d, std_rest_days_60d, back_to_back_7d

#### Fatigue signals (EWM + deltas)

- vel_ewm_7d, vel_ewm_14d, vel_delta_7d_vs_30d, vel_delta_14d_vs_30d
- spin_ewm_7d, spin_ewm_14d, spin_delta_7d_vs_30d, spin_delta_14d_vs_30d
- Optional: vel_volatility_5g, spin_volatility_5g

#### Pitch Mix

- %breaking_7d, %breaking_14d, delta_%breaking_7d_vs_30d

#### Role & History

- role_starter_flag, avg_ip_per_app_60d
- prior_il_count_2y, age_bucket

#### Data Quality

- data_completeness_score; per-family missingness flags

### Rules

All features computed with data ≤ as_of_date.

EWMs: choose half-life in appearances (e.g., 3–5 appearances).

### Success checks

- backend/etl/features_spec.md updated with formulas and half-life choices.
- docs/feature_coverage.md: coverage % per feature per season; any feature < 70% coverage is flagged.

## 4) Rolling-Origin Backtests (time-aware)

### Plan

- Block A: Train 2022 → Validate Q1 2023
- Block B: Train 2022–Q1 2023 → Validate Q2 2023
- … continue quarterly to cover 2023 and 2024 (at least 6 blocks).

### Metrics (per block)

- PR-AUC (primary for imbalance)
- Recall@Top-10% and Lift@Top-10% (decision utility)
- AUC (secondary)
- Brier score or ECE (calibration)
- Subgroups: starters vs relievers

### Success checks

docs/backtest_report.md table with rows per block:

Train window, Val window, Pos rate (train/val), AUC, PR-AUC, Recall@10%, Lift@10%, Brier.

A clear winner selection rule: promote only if PR-AUC and Recall@10% improve and calibration (Brier/ECE) not worse.

## 5) Model & Calibration (stable, interpretable)

### XGBoost defaults

- max_depth=4–6, learning_rate=0.05–0.1, n_estimators=400–1000
- subsample=0.8, colsample_bytree=0.8
- eval_metric='aucpr'
- scale_pos_weight = (neg/pos) computed per training window
- Early stopping on each validation block

### Calibration

Hold out most-recent block (never used for tuning) for isotonic calibration.

Subgroup calibrators: one for starters, one for relievers.

### Success checks

- Reliability curves saved per block in /artifacts/calibration/.
- Post-calibration Brier improves vs pre-calibration on the holdout.

## 6) Payload Upgrade (trust & actionability)

### Add to /risk response

- model_version: e.g., xgb_v2_2025-08-21
- risk_score_calibrated: 0–1
- risk_bucket: low/med/high via dynamic top-K (e.g., top 10% per active staff today)
- confidence: low/med/high from data_completeness_score
- contributors: top-3 {name, value, direction, cohort_percentile}
- cohort_percentile: among same role and age bucket
- recommended_actions: up to 3 (rest, workload cap, mech review)
- data_completeness: map of booleans by family

### Success checks

- Frontend renders bucket, contributors, cohort percentile, and confidence.
- Internal QA: contributors match the sign/direction of the underlying feature values.

## 7) Data Quality Gates (prevent silent regressions)

Add a lightweight test script that prints:

- Row counts per table and season (pitches/appearances/injuries/features).
- % missing for release_speed, release_spin_rate (seasonal).
- Monotonic sanity: more rest → lower predicted risk on average; larger negative vel_delta_7d_vs_30d → higher predicted risk on average. (Not enforced strictly, but report.)

### Success checks

Store outputs in docs/data_quality_gates.md (updated every run).

If a gate fails, model training should abort with a helpful message.

## 8) Performance & Serving

### SQLite dev improvements

Add the indices above; VACUUM analyze after big loads.

Cache the latest features snapshot per pitcher for fast /risk responses.

### Postgres prep (optional, recommended)

Add docker-compose-postgres.yml.

Schema identical to SQLite but with:

```sql
CREATE INDEX IF NOT EXISTS pitches_date_pid_idx ON pitches (game_date, pitcher_id);
CREATE INDEX IF NOT EXISTS appearances_date_pid_idx ON appearances (game_date, pitcher_id);
```

Plan partitioning by season later.

### Success checks

- /risk 95p latency < 150ms locally for warm cache.
- /workload returns appearance series for real pitchers over 30–60 days within 300ms locally.

## 9) Weekly Metrics Page (one source of truth)

Expose /metrics (or generate a markdown) with:

- Train/Val windows
- Pos rate (train/val)
- AUC, PR-AUC, Recall@Top-10%, Lift@Top-10%, Brier
- Top 10 features by gain (overall + by subgroup)
- Calibration decile table (predicted vs observed)
- Drift snapshot: mean velocity, mean spin, mean roll7d_pitch_count vs prior week; PSI for 3–5 key features

### Success checks

Add docs/weekly_model_report_TEMPLATE.md and publish one real instance after the first full run.

## 10) Frontend polish (fast wins)

### On Staff Overview:

- Add Risk Percentile and Confidence columns.
- Add a "signal chips" line (e.g., "−2.3 MPH / Short Rest / +BB%").

### On Pitcher Detail:

- Add risk sparkline (last 30 days).
- Add contributors panel with inline definitions (hover tooltip explaining each signal).

### Success checks

A coach can answer in <30s: "Who's top-risk today, and why?"

## 11) Deliverables for this sprint

- docs/backtest_report.md with multi-block results.
- docs/label_report.md + docs/injury_filters.md (regex lists).
- docs/feature_coverage.md & docs/data_quality_gates.md.
- Calibrated model artifacts in /artifacts/… with model_version.
- /risk enriched payload visible in UI.

## 12) Risks & Mitigations

- Label sparsity: Use 2022–2024 to lift positives; keep PR-AUC the north star.
- Noisy velocity/spin: EWM + deltas; minimum sample thresholds; completeness flags.
- Data drift: Weekly metrics + PSI checks; monthly retrain cadence.

## 13) Promotion Gate (when to call it "ready")

vs current model, aggregated across backtest blocks:

- PR-AUC: + ≥10% relative
- Recall@Top-10%: + ≥10% relative
- Brier: not worse (or improved)
- Subgroup (starter/reliever) calibration ECE ≤ 0.05.
- Latency within targets; payload fields complete.

## TL;DR (do these first)

1. Load 2022–2024 (pitches + appearances) with idempotent ETL.
2. Clean IL labels (arm-related only) and generate per-appearance snapshots with 21-day forward labels.
3. Compute ≤40 leak-free features (acute/chronic, EWM deltas, rest, mix, role/history).
4. Run rolling backtests; train XGB with scale_pos_weight; isotonic calibration + role sub-calibrators.
5. Upgrade /risk payload (bucket, contributors, cohort percentile, confidence, actions).

