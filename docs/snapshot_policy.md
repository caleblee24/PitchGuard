# 📋 Snapshot Policy & Evaluation Universe

**PitchGuard Enhanced Model Validation**  
*Version: 1.0* | *Date: August 21, 2025*

---

## 🎯 **EVALUATION UNIVERSE DEFINITION**

### **Snapshot Unit: Per-Appearance**
- **Definition**: Each row represents a pitcher's appearance on a specific `game_date`
- **Rationale**: Aligns with coaching decisions and workload management
- **Alternative Considered**: Per-pitch (too granular) vs Per-game (too coarse)

### **Evaluation Universe: Full Background Cohort**
- **Scope**: All per-appearance snapshots league-wide in 2022-2024 regular seasons
- **Inclusion**: Every eligible pitcher-day, including non-injury appearances
- **Exclusion**: Post-season, spring training, exhibition games

### **Risk Horizon: 21-Day Window**
- **Label Definition**: `1` if any IL start in `(game_date, game_date + 21 days]`
- **Rationale**: Captures acute injury risk while providing actionable lead time
- **Alternative Considered**: 14 days (too short) vs 30 days (too long)

---

## 🏥 **INJURY FILTERING POLICY**

### **Included Injury Types**
- **Arm-Related**: Elbow, forearm, shoulder, lat, rotator cuff, UCL
- **Specific Conditions**: Strain, inflammation, soreness for arm areas
- **Examples**: 
  - "Right elbow inflammation"
  - "Left shoulder strain"
  - "UCL sprain"
  - "Forearm tightness"

### **Excluded Injury Types**
- **Non-Arm**: Illness, COVID, concussion, HBP fractures
- **Non-Medical**: Paternity, bereavement, personal reasons
- **Examples**:
  - "COVID-19"
  - "Concussion"
  - "Paternity leave"
  - "Personal reasons"

### **Pre-Injury Blackout: 3-Day Window**
- **Policy**: If IL starts within 0-3 days of appearance, exclude from training
- **Rationale**: Avoids trivial near-event leakage
- **Reporting**: Track separately as "near-event" performance

---

## 📊 **LABEL GENERATION PROCESS**

### **Step 1: Data Collection**
```sql
-- Collect all appearances
SELECT 
    pitcher_id,
    game_date,
    team,
    role,
    pitches_thrown,
    innings_pitched
FROM appearances 
WHERE game_date BETWEEN '2022-01-01' AND '2024-12-31'
  AND season_type = 'regular'
```

### **Step 2: Injury Labeling**
```sql
-- Label injuries within 21-day window
SELECT 
    a.pitcher_id,
    a.game_date,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM injuries i 
            WHERE i.pitcher_id = a.pitcher_id
              AND i.injury_date BETWEEN a.game_date + 1 AND a.game_date + 21
              AND i.injury_type IN ('elbow', 'shoulder', 'forearm', 'lat', 'rotator_cuff', 'UCL')
        ) THEN 1
        ELSE 0
    END as injury_within_21d
FROM appearances a
```

### **Step 3: Blackout Application**
```sql
-- Apply 3-day blackout
SELECT 
    pitcher_id,
    game_date,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM injuries i 
            WHERE i.pitcher_id = a.pitcher_id
              AND i.injury_date BETWEEN a.game_date AND a.game_date + 3
        ) THEN NULL  -- Exclude from training
        ELSE injury_within_21d
    END as training_label
FROM labeled_appearances
```

---

## 📈 **POSITIVE RATE TARGETS**

### **Expected Positive Rates by Season**
- **2022**: 2.1% (baseline)
- **2023**: 2.3% (increasing trend)
- **2024**: 2.0% (partial season)

### **By Role**
- **Starters**: 2.8% (higher workload)
- **Relievers**: 1.5% (lower workload)

### **By Age Bucket**
- **< 26**: 1.8% (younger, more resilient)
- **26-30**: 2.2% (peak performance)
- **> 30**: 2.6% (aging, higher risk)

---

## 🔍 **QUALITY ASSURANCE**

### **Manual Validation Protocol**
- **Sample Size**: 25 positive cases per season
- **Validation Criteria**: ≥90% must be arm-related injuries
- **Review Process**: Medical expert review of injury descriptions

### **Data Quality Checks**
- **Coverage**: ≥95% of appearances have complete data
- **Consistency**: Injury dates align with IL stint records
- **Completeness**: All required features available for ≥80% of snapshots

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Phase 1: Policy Definition** ✅
- [x] Define snapshot unit (per-appearance)
- [x] Define risk horizon (21 days)
- [x] Define injury inclusion/exclusion criteria
- [x] Define blackout policy (3 days)

### **Phase 2: Data Processing**
- [ ] Generate full background cohort (all appearances)
- [ ] Apply injury labeling logic
- [ ] Apply blackout exclusions
- [ ] Validate positive rates by season/role

### **Phase 3: Quality Assurance**
- [ ] Manual validation of 25 positive cases
- [ ] Data quality checks
- [ ] Coverage analysis
- [ ] Consistency validation

---

## 📊 **EXPECTED OUTCOMES**

### **Cohort Statistics**
- **Total Snapshots**: ~150,000 (2022-2024)
- **Positive Cases**: ~3,000 (2% positive rate)
- **Training Set**: ~120,000 (80% for training)
- **Validation Set**: ~30,000 (20% for validation)

### **Performance Targets**
- **PR-AUC**: >0.70 (vs full background cohort)
- **Recall@Top-10%**: >0.40
- **Precision@Top-10%**: >0.15
- **Brier Score**: <0.20

---

*This policy ensures our validation is defensible and represents real-world performance against the full population of pitcher appearances, not just injury cases.*
