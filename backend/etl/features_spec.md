# 🔧 Feature Specification & Contract

**PitchGuard Enhanced Model**  
*Version: 2.0* | *Date: August 21, 2025*

---

## 🎯 **FEATURE CONTRACT OVERVIEW**

### **Purpose**
- **Single Source of Truth**: All feature definitions, types, and formulas
- **Training/Serving Parity**: Guaranteed consistency between training and inference
- **Version Control**: Tracked feature evolution and compatibility
- **Quality Assurance**: Coverage and variance requirements

### **Feature Fingerprint**
- **Definition**: SHA256 hash of sorted feature names
- **Usage**: Fail-closed validation at serve time
- **Storage**: Saved with model artifacts and logged with requests

---

## 📊 **CORE FEATURE SET (32 Features)**

### **1. Workload Features (12 features)**

#### **Pitch Count Metrics**
| Feature | Type | Unit | Formula | Window | Min Samples | Missing Flag |
|---------|------|------|---------|--------|-------------|--------------|
| `pitch_count_7d` | float | pitches | Sum of pitches in last 7 days | 7d | 1 | `pitch_count_7d_missing` |
| `pitch_count_14d` | float | pitches | Sum of pitches in last 14 days | 14d | 1 | `pitch_count_14d_missing` |
| `pitch_count_30d` | float | pitches | Sum of pitches in last 30 days | 30d | 1 | `pitch_count_30d_missing` |
| `avg_pitches_per_appearance` | float | pitches | Mean pitches per appearance | 30d | 3 | `avg_pitches_missing` |

#### **Rest & Recovery**
| Feature | Type | Unit | Formula | Window | Min Samples | Missing Flag |
|---------|------|------|---------|--------|-------------|--------------|
| `rest_days` | int | days | Days since last appearance | Current | 1 | `rest_days_missing` |
| `avg_rest_days` | float | days | Mean rest days between appearances | 30d | 3 | `avg_rest_missing` |
| `short_rest_appearances` | int | count | Appearances with <3 days rest | 30d | 1 | `short_rest_missing` |

#### **Workload Intensity**
| Feature | Type | Unit | Formula | Window | Min Samples | Missing Flag |
|---------|------|------|---------|--------|-------------|--------------|
| `workload_intensity_7d` | float | score | (pitch_count_7d * 0.7) + (appearances_7d * 30 * 0.3) | 7d | 1 | `intensity_7d_missing` |
| `workload_intensity_14d` | float | score | (pitch_count_14d * 0.7) + (appearances_14d * 30 * 0.3) | 14d | 1 | `intensity_14d_missing` |
| `high_workload_days` | int | count | Days with >100 pitches | 30d | 1 | `high_workload_missing` |

### **2. Velocity Features (8 features)**

#### **Velocity Trends**
| Feature | Type | Unit | Formula | Window | Min Samples | Missing Flag |
|---------|------|------|---------|--------|-------------|--------------|
| `avg_velocity_7d` | float | mph | Mean velocity in last 7 days | 7d | 10 | `vel_7d_missing` |
| `avg_velocity_14d` | float | mph | Mean velocity in last 14 days | 14d | 20 | `vel_14d_missing` |
| `velocity_decline_7d` | float | mph | (avg_vel_7d - avg_vel_14d) | 14d | 20 | `vel_decline_missing` |
| `velocity_decline_14d` | float | mph | (avg_vel_14d - avg_vel_30d) | 30d | 30 | `vel_decline_14d_missing` |

#### **Velocity Stability**
| Feature | Type | Unit | Formula | Window | Min Samples | Missing Flag |
|---------|------|------|---------|--------|-------------|--------------|
| `velocity_std_7d` | float | mph | Standard deviation of velocity | 7d | 10 | `vel_std_7d_missing` |
| `velocity_std_14d` | float | mph | Standard deviation of velocity | 14d | 20 | `vel_std_14d_missing` |
| `velocity_trend` | float | slope | Linear trend coefficient | 30d | 30 | `vel_trend_missing` |

### **3. Spin Rate Features (6 features)**

#### **Spin Rate Metrics**
| Feature | Type | Unit | Formula | Window | Min Samples | Missing Flag |
|---------|------|------|---------|--------|-------------|--------------|
| `avg_spin_rate_7d` | float | rpm | Mean spin rate in last 7 days | 7d | 10 | `spin_7d_missing` |
| `avg_spin_rate_14d` | float | rpm | Mean spin rate in last 14 days | 14d | 20 | `spin_14d_missing` |
| `spin_rate_decline_7d` | float | rpm | (avg_spin_7d - avg_spin_14d) | 14d | 20 | `spin_decline_missing` |
| `spin_rate_std_7d` | float | rpm | Standard deviation of spin rate | 7d | 10 | `spin_std_7d_missing` |

#### **Spin Rate Trends**
| Feature | Type | Unit | Formula | Window | Min Samples | Missing Flag |
|---------|------|------|---------|--------|-------------|--------------|
| `spin_rate_trend` | float | slope | Linear trend coefficient | 30d | 30 | `spin_trend_missing` |
| `spin_velocity_ratio` | float | ratio | avg_spin_rate / avg_velocity | 14d | 20 | `spin_vel_ratio_missing` |

### **4. Pitch Mix Features (6 features)**

#### **Pitch Type Distribution**
| Feature | Type | Unit | Formula | Window | Min Samples | Missing Flag |
|---------|------|------|---------|--------|-------------|--------------|
| `fastball_pct` | float | % | Percentage of fastballs | 30d | 50 | `fastball_pct_missing` |
| `breaking_pct` | float | % | Percentage of breaking balls | 30d | 50 | `breaking_pct_missing` |
| `off_speed_pct` | float | % | Percentage of off-speed | 30d | 50 | `off_speed_pct_missing` |

#### **Pitch Mix Stability**
| Feature | Type | Unit | Formula | Window | Min Samples | Missing Flag |
|---------|------|------|---------|--------|-------------|--------------|
| `pitch_mix_change_7d` | float | % | Change in fastball percentage | 7d | 20 | `mix_change_missing` |
| `pitch_mix_change_14d` | float | % | Change in fastball percentage | 14d | 30 | `mix_change_14d_missing` |
| `pitch_mix_volatility` | float | std | Standard deviation of fastball % | 30d | 50 | `mix_volatility_missing` |

---

## 🔧 **FEATURE ENGINEERING FORMULAS**

### **Workload Intensity Calculation**
```python
def calculate_workload_intensity(pitch_count, appearances, days):
    """Calculate workload intensity score."""
    avg_pitches = pitch_count / max(appearances, 1)
    frequency = appearances / max(days, 1)
    intensity = (avg_pitches * 0.7) + (frequency * 30 * 0.3)
    return intensity
```

### **Velocity Trend Calculation**
```python
def calculate_velocity_trend(velocities, dates):
    """Calculate linear trend in velocity."""
    if len(velocities) < 2:
        return 0.0
    
    x = np.arange(len(velocities))
    y = np.array(velocities)
    slope = np.polyfit(x, y, 1)[0]
    return slope
```

### **Pitch Mix Change**
```python
def calculate_pitch_mix_change(current_pct, baseline_pct):
    """Calculate change in pitch mix percentage."""
    if baseline_pct == 0:
        return 0.0
    return ((current_pct - baseline_pct) / baseline_pct) * 100
```

---

## 📊 **FEATURE COVERAGE REQUIREMENTS**

### **Minimum Coverage Thresholds**
- **Core Features**: ≥80% coverage required
- **Velocity Features**: ≥70% coverage required (some pitchers lack velocity data)
- **Spin Rate Features**: ≥60% coverage required (less common)
- **Pitch Mix Features**: ≥75% coverage required

### **Variance Requirements**
- **All Features**: Non-zero variance required
- **Top-10 Features**: ≥0.1 variance required
- **Categorical Features**: ≥2 unique values required

### **Missing Data Handling**
- **Imputation**: Median imputation for missing values
- **Flags**: Missing flags for all features
- **Exclusion**: Snapshots with >50% missing features excluded

---

## 🔍 **FEATURE FINGERPRINT SYSTEM**

### **Fingerprint Generation**
```python
def generate_feature_fingerprint(feature_names):
    """Generate SHA256 hash of sorted feature names."""
    sorted_features = sorted(feature_names)
    feature_string = "|".join(sorted_features)
    return hashlib.sha256(feature_string.encode()).hexdigest()
```

### **Fingerprint Storage**
- **Training**: Saved with model artifact as `feature_fingerprint.txt`
- **Serving**: Logged with each risk assessment request
- **Validation**: Fail-closed if fingerprints don't match

### **Version Control**
- **Feature Versions**: Tracked in `features_spec.md`
- **Compatibility**: Backward compatibility requirements
- **Migration**: Automated feature migration scripts

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Phase 1: Feature Definition** ✅
- [x] Define all 32 features with formulas
- [x] Specify data types and units
- [x] Define coverage requirements
- [x] Create missing data handling

### **Phase 2: Fingerprint System**
- [ ] Implement fingerprint generation
- [ ] Add fingerprint validation to serving
- [ ] Create fingerprint logging
- [ ] Test fingerprint consistency

### **Phase 3: Quality Assurance**
- [ ] Validate feature coverage
- [ ] Check feature variance
- [ ] Test missing data handling
- [ ] Verify training/serving parity

---

## 📊 **EXPECTED OUTCOMES**

### **Feature Coverage**
- **Overall Coverage**: ≥85% across all features
- **Core Features**: ≥90% coverage
- **Velocity Features**: ≥75% coverage
- **Spin Rate Features**: ≥65% coverage

### **Performance Impact**
- **Training Time**: <5 minutes for full dataset
- **Inference Time**: <100ms per prediction
- **Memory Usage**: <2GB for model + features
- **Accuracy**: Maintain or improve current performance

---

*This specification ensures training/serving parity and provides a clear contract for feature engineering across the entire system.*
