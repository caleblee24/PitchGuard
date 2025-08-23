# Data Dictionary

## Overview

This document defines all data structures, field definitions, and constraints for the PitchGuard system. The data model follows a hierarchical structure from granular pitch data to aggregated features and predictions.

## Core Tables

### 1. Pitches Table

**Purpose**: Raw pitch-level data from MLB games
**Granularity**: One row per pitch thrown
**Source**: Baseball Savant API or mock data generator

| Field | Type | Description | Constraints | Example |
|-------|------|-------------|-------------|---------|
| `game_date` | DATE | Date of the game | NOT NULL, >= 2015-01-01 | 2024-04-15 |
| `pitcher_id` | INTEGER | Unique pitcher identifier | NOT NULL, > 0 | 12345 |
| `pitch_type` | VARCHAR(10) | Type of pitch thrown | NOT NULL, in valid types | 'FF', 'SL', 'CH' |
| `release_speed` | DECIMAL(4,1) | Pitch velocity in MPH | NULL allowed, 50-110 | 94.5 |
| `release_spin_rate` | INTEGER | Spin rate in RPM | NULL allowed, 0-3500 | 2450 |
| `pitch_number` | INTEGER | Pitch number in at-bat | NOT NULL, > 0 | 3 |
| `at_bat_id` | BIGINT | Unique at-bat identifier | NOT NULL | 123456789 |

**Valid Pitch Types**: FF (Four-seam), SL (Slider), CH (Changeup), CT (Cutter), CB (Curveball), SI (Sinker), KC (Knuckle-curve), FS (Splitter), FO (Forkball), KN (Knuckleball)

### 2. Appearances Table

**Purpose**: Aggregated data per pitcher per game appearance
**Granularity**: One row per pitcher per game
**Derived from**: Pitches table aggregation

| Field | Type | Description | Constraints | Example |
|-------|------|-------------|-------------|---------|
| `game_date` | DATE | Date of the game | NOT NULL | 2024-04-15 |
| `pitcher_id` | INTEGER | Unique pitcher identifier | NOT NULL | 12345 |
| `pitches_thrown` | INTEGER | Total pitches in appearance | NOT NULL, > 0 | 85 |
| `avg_vel` | DECIMAL(4,1) | Average velocity across all pitches | NULL allowed | 93.2 |
| `avg_spin` | INTEGER | Average spin rate across all pitches | NULL allowed | 2350 |
| `outs_recorded` | INTEGER | Outs recorded in appearance | NULL allowed, >= 0 | 15 |
| `innings_pitched` | DECIMAL(3,1) | Innings pitched (fractional) | NULL allowed, >= 0 | 5.0 |

### 3. Injuries Table

**Purpose**: Track IL (Injured List) stints for pitchers
**Granularity**: One row per IL stint
**Source**: MLB transaction data or mock data

| Field | Type | Description | Constraints | Example |
|-------|------|-------------|-------------|---------|
| `pitcher_id` | INTEGER | Unique pitcher identifier | NOT NULL | 12345 |
| `il_start` | DATE | Date IL stint began | NOT NULL | 2024-05-20 |
| `il_end` | DATE | Date IL stint ended | NULL allowed | 2024-07-15 |
| `stint_type` | VARCHAR(50) | Type of injury/IL stint | NOT NULL | 'Elbow inflammation' |

**Common Stint Types**: Elbow inflammation, Shoulder strain, Back tightness, Forearm strain, Tommy John surgery, Rotator cuff, Oblique strain, Hamstring strain

### 4. Feature Snapshots Table

**Purpose**: Computed features for model training and prediction
**Granularity**: One row per pitcher per date
**Derived from**: Appearances and Injuries tables

| Field | Type | Description | Constraints | Example |
|-------|------|-------------|-------------|---------|
| `as_of_date` | DATE | Date of feature snapshot | NOT NULL | 2024-04-15 |
| `pitcher_id` | INTEGER | Unique pitcher identifier | NOT NULL | 12345 |
| `roll3g_pitch_count` | INTEGER | Sum of pitches over last 3 games | >= 0 | 245 |
| `roll3d_pitch_count` | INTEGER | Sum of pitches over last 3 days | >= 0 | 85 |
| `roll7d_pitch_count` | INTEGER | Sum of pitches over last 7 days | >= 0 | 180 |
| `roll14d_pitch_count` | INTEGER | Sum of pitches over last 14 days | >= 0 | 320 |
| `avg_vel_7d` | DECIMAL(4,1) | Average velocity over last 7 days | NULL allowed | 93.5 |
| `vel_drop_vs_30d` | DECIMAL(4,1) | 7d avg vel - 30d avg vel | NULL allowed | -1.2 |
| `avg_spin_7d` | INTEGER | Average spin rate over last 7 days | NULL allowed | 2350 |
| `spin_drop_vs_30d` | INTEGER | 7d avg spin - 30d avg spin | NULL allowed | -50 |
| `rest_days` | INTEGER | Days since last appearance | >= 0 | 4 |
| `label_injury_within_21d` | BOOLEAN | Injury within 21 days (target) | NOT NULL | FALSE |
| `data_completeness` | VARCHAR(10) | Completeness level | in ['high','med','low'] | 'high' |

### 5. Model Registry Table

**Purpose**: Track trained models and their performance
**Granularity**: One row per model version

| Field | Type | Description | Constraints | Example |
|-------|------|-------------|-------------|---------|
| `model_name` | VARCHAR(100) | Model identifier | NOT NULL | 'logistic_regression_v1' |
| `created_at` | TIMESTAMP | Model creation timestamp | NOT NULL | 2024-04-15 10:30:00 |
| `framework` | VARCHAR(50) | ML framework used | NOT NULL | 'scikit-learn' |
| `metrics` | JSONB | Performance metrics | NOT NULL | {"auc": 0.78, "precision": 0.65} |
| `artifact_ref` | VARCHAR(255) | Path to model file | NOT NULL | '/models/logistic_v1.pkl' |

## Derived Fields & Computations

### Feature Engineering Rules

1. **Rolling Windows**: All rolling calculations use only data from dates <= as_of_date (no future leakage)

2. **Velocity/Spin Averages**: 
   - Require minimum 20 pitches in window for reliable average
   - Mark as NULL if insufficient data
   - Use weighted average by pitch count if multiple appearances

3. **Data Completeness Levels**:
   - **High**: ≥20 pitches in last 7 days, ≥5 appearances in last 30 days
   - **Medium**: ≥10 pitches in last 7 days, ≥3 appearances in last 30 days  
   - **Low**: <10 pitches in last 7 days or <3 appearances in last 30 days

4. **Rest Days Calculation**:
   - Days between consecutive appearances
   - 0 for first appearance of season
   - NULL if no previous appearance in dataset

### Labeling Rules

1. **Injury Window**: 21-day forward-looking window (as_of_date + 1 to as_of_date + 21)

2. **Positive Label**: Any IL stint starting within the 21-day window

3. **Multiple Stints**: Use earliest start date if multiple stints in window

4. **Leakage Prevention**: Only use IL data available as of as_of_date

## Data Quality Constraints

### Validation Rules

1. **Pitch Data**:
   - release_speed: 50-110 MPH (reasonable MLB range)
   - release_spin_rate: 0-3500 RPM
   - pitch_number: > 0 within at_bat
   - game_date: >= 2015-01-01 (Statcast era)

2. **Appearance Data**:
   - pitches_thrown: > 0
   - avg_vel: 50-110 MPH if not NULL
   - avg_spin: 0-3500 RPM if not NULL
   - innings_pitched: >= 0

3. **Feature Data**:
   - All rolling counts: >= 0
   - Velocity drops: reasonable range (-10 to +5 MPH)
   - Spin drops: reasonable range (-500 to +500 RPM)
   - rest_days: >= 0

### Data Completeness Requirements

- **Training**: Minimum 1000 feature snapshots with positive labels
- **Prediction**: Minimum 3 appearances in last 30 days for reliable risk assessment
- **Missing Data**: Handle gracefully with NULL values and completeness indicators

## API Response Formats

### Risk Assessment Response

```json
{
  "risk_score": 0.23,
  "risk_percentage": 23,
  "risk_level": "medium",
  "top_contributors": [
    {
      "name": "High pitch count",
      "value": 245,
      "direction": "increasing",
      "explanation": "Pitcher threw 245 pitches in last 3 games"
    }
  ],
  "data_completeness": "high",
  "confidence": 0.85
}
```

### Workload History Response

```json
{
  "pitcher_id": 12345,
  "workload_series": [
    {
      "date": "2024-04-15",
      "pitch_count": 85,
      "avg_vel": 93.2,
      "avg_spin": 2350
    }
  ]
}
```

## Mock Data Specifications

### Sample Data Requirements

1. **3 Pitchers** with varying patterns:
   - Pitcher A: Normal workload, no injury
   - Pitcher B: High workload, velocity decline, injury
   - Pitcher C: Low workload, consistent performance

2. **Time Range**: 3 months of data (April-June 2024)

3. **Realistic Patterns**:
   - 80-120 pitches per appearance
   - 90-98 MPH velocity range
   - 2000-2800 RPM spin rate range
   - 3-5 days between appearances

4. **Injury Scenario**: One pitcher with velocity decline (-2 to -3 MPH) followed by injury 10-15 days later
