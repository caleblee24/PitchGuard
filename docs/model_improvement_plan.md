# PitchGuard — Model Improvement Plan (One Pager)

## 1) Objective

Deliver well-calibrated, time-aware injury risk for MLB pitchers within 21 days, optimized for operational utility (who to flag today) rather than generic accuracy.

## 2) Dataset & Label Policy

**Scope**: MLB pitchers, last 3–5 seasons.

**Label (positive)**: IL stint start in (as_of_date, as_of_date+21d].

**Exclusions (when possible)**: Non–overuse events (e.g., illness/trauma). Keep elbow/shoulder.

**Cold start**: If <2 appearances in last 21d → defer to cohort prior + low confidence flag.

**Data integrity rules**:
- No future leakage: features computed only from data ≤ as_of_date.
- Minimum data: velocity/spin features require ≥20 pitches in last 7d; else mark missing + add missingness flags.

## 3) Evaluation Protocol (commit to this)

**Split**: Rolling-origin backtests (e.g., train thru Jun, test Jul; slide monthly).

**Primary metrics**:
- AUC (discrimination)
- PR-AUC (class imbalance)
- Recall@Top-10% (ops utility)
- Lift@Top-10% vs random 10%
- Brier score + calibration plot

**Subgroup reporting**: Starters vs relievers (and optionally age buckets).

**Promotion gate**: New model must win on PR-AUC and Recall@Top-10%, and be calibrated (Brier ↓ or ECE ↓).

## 4) Feature Roadmap (add in this order)

### Role & workload context
- Starter/reliever, back-to-back days, multi-inning in last 7/14d.
- Acute vs chronic pitch counts: last game, last 3 games, 7/14/30d.

### Fatigue signals
- Velo & spin EWMs; deltas vs 30d baseline; volatility (std dev).

### Pitch mix & shape (lightweight)
- % breaking balls last 7/14d; delta vs 30d.
- Release point variance; extension trend (if available).

### History & aging
- Prior IL count (2 yrs), age bucket, year-to-date innings vs same date last year.

**Keep total features ≤40; include missingness flags per family.**

## 5) Modeling Roadmap

**Baseline**: Logistic Regression + Platt/Isotonic calibration on held-out time block.

**Upgrade**: XGBoost/LightGBM with:
- Class weights for imbalance
- (Optional) Monotonic constraints (e.g., more rest ↓ risk; larger velo drop ↑ risk)

**Sequence awareness (lightweight)**: add lagged snapshots (t-7, t-14) for key deltas & risk.

## 6) Calibration Policy

- Always ship calibrated probabilities.
- Fit calibration on most recent hold-out window (never on training folds).
- Subgroup calibration: separate calibrators for starters vs relievers.

## 7) Payload Enhancements (returned by /risk)

- `model_version`
- `risk_score_calibrated` (0–1) + `risk_bucket` (dynamic top-K)
- `confidence` (low/med/high) from data completeness
- `contributors` (top 3 signals: name, value, direction, cohort percentile)
- `cohort_percentile` (risk vs role/age cohort)
- `recommended_actions` (short, non-deterministic guidance)
- `data_completeness` map (per feature family)

## 8) Monitoring & Retraining

**Daily drift checks**: feature PSI for pitch count, velo, spin; population mean risk; % high-risk flags.

**Triggers**: PSI > 0.2 or ≥20% change in population risk mean → schedule retrain.

**Cadence**: Monthly retrain on rolling 2–3 yr window; refresh calibration each cycle.

## 9) Experiment Discipline

- One change per experiment; log experiment_id, feature set, dates, metrics.
- Maintain a small leaderboard: e.g., B0 (logistic), X1 (XGB v1), X2 (+mix), X3 (+EWMs), etc.
- Promote only if it wins and remains calibrated.

## 10) Two-Sprint Plan (concrete)

### Sprint 1 (7–10 days)
- Add features: role, acute/chronic loads, pitch-mix deltas, EWMs.
- Swap to XGBoost + isotonic calibration; subgroup calibration by role.
- Ship enriched payload (confidence, contributors, cohort_percentile).
- Run rolling backtests; publish baseline vs new: AUC, PR-AUC, Recall@Top-10%, Brier.

### Sprint 2 (7–10 days)
- Add release-point variance/extension trend (if available).
- Add history features (prior IL, age, YTD innings vs LY).
- Introduce lagged (t-7, t-14) snapshot features.
- Pilot discrete-time hazard framing; compare Peak-risk-in-window vs single-point risk.

## 11) Weekly Model Report (template)

```
Model: <version> | Data window: <dates>
Backtest windows: list

AUC / PR-AUC / Recall@Top-10% / Lift@10% / Brier (overall + by role)

Calibration plot link; ECE or Brier deltas vs prior model

Drift: PSI per key feature; population risk mean/std

Top contributors (avg): e.g., vel_drop_vs_30d, roll7d_pitch_count, rest_days

False-positive review: 3 examples (why flagged)

Action items: next-feature adds, data gaps, retrain decision
```

## 12) Acceptance Criteria for "Production" (MVP scope)

- Rolling backtests show +≥15% Lift@Top-10% vs baseline and no worse calibration (Brier ≤ baseline).
- Subgroup calibration errors within acceptable range (e.g., ECE ≤ 0.05) for both starters/relievers.
- Payload includes contributors, confidence, cohort percentile, and recommended actions.

## 13) Risks & Mitigations

- **Label noise (non-overuse IL)**: filter where feasible; otherwise soften claims and rely on calibration.
- **Sparse data (relievers)**: heavier use of EWMs and cohort priors; explicit low-confidence flag.
- **Drift from league trends**: monthly retrains + PSI monitoring.

---

**Owner(s)**: TBD  
**Current version**: MVP Baseline (Logistic Regression)  
**Next retrain date**: TBD  
**Last updated**: 2025-08-20
