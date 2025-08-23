# Modeling Plan

## Overview

The PitchGuard injury risk prediction model uses a time-aware logistic regression approach with probability calibration to predict 21-day injury risk. The model prioritizes interpretability and actionable insights over complex black-box predictions.

## Model Architecture

### Base Model: Logistic Regression

**Rationale:**
- **Interpretability**: Coefficients directly show feature importance and direction
- **Calibration**: Natural probability outputs that can be calibrated
- **Stability**: Less prone to overfitting on limited injury data
- **Explainability**: Easy to generate "why now" explanations

### Model Components

```python
class InjuryRiskModel:
    def __init__(self):
        # Feature preprocessing
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_classif, k=10)
        
        # Base classifier
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Probability calibration
        self.calibrator = CalibratedClassifierCV(
            self.classifier,
            method='isotonic',
            cv=5
        )
        
        # Feature importance tracking
        self.feature_names = None
        self.feature_importance = None
```

## Feature Engineering

### Core Features

1. **Rolling Workload Features**
   - `roll3g_pitch_count`: Sum of pitches over last 3 games
   - `roll7d_pitch_count`: Sum of pitches over last 7 days
   - `roll14d_pitch_count`: Sum of pitches over last 14 days
   - `roll30d_pitch_count`: Sum of pitches over last 30 days

2. **Velocity Trend Features**
   - `avg_vel_7d`: Average velocity over last 7 days
   - `vel_drop_vs_30d`: 7-day avg velocity - 30-day avg velocity
   - `vel_std_7d`: Standard deviation of velocity over last 7 days

3. **Spin Rate Features**
   - `avg_spin_7d`: Average spin rate over last 7 days
   - `spin_drop_vs_30d`: 7-day avg spin - 30-day avg spin
   - `spin_std_7d`: Standard deviation of spin rate over last 7 days

4. **Recovery Features**
   - `rest_days`: Days since last appearance
   - `avg_rest_days_30d`: Average rest days over last 30 days
   - `rest_pattern`: Pattern of rest days (consistent vs variable)

5. **Workload Intensity Features**
   - `pitches_per_inning`: Average pitches per inning over last 7 days
   - `high_intensity_games`: Number of games with >100 pitches in last 30 days
   - `consecutive_high_workload`: Consecutive games with >90 pitches

### Feature Selection

```python
def select_features(X, y, k=10):
    """Select top k features using ANOVA F-test."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()]
    
    return X_selected, selected_features, selector
```

### Feature Scaling

```python
def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
```

## Training Strategy

### Time-Based Split

**Rationale:** Prevent data leakage by ensuring no future information is used in training

```python
def create_time_based_split(features_df, train_end_date, val_start_date):
    """Create time-based train/validation split."""
    
    # Training data: All data up to train_end_date
    train_data = features_df[features_df['as_of_date'] <= train_end_date]
    
    # Validation data: Data from val_start_date onwards
    val_data = features_df[features_df['as_of_date'] >= val_start_date]
    
    # Ensure no overlap
    assert train_data['as_of_date'].max() < val_data['as_of_date'].min()
    
    return train_data, val_data
```

### Cross-Validation Strategy

```python
def time_series_cv_split(features_df, n_splits=5):
    """Time series cross-validation splits."""
    from sklearn.model_selection import TimeSeriesSplit
    
    # Sort by date
    features_df = features_df.sort_values('as_of_date')
    
    # Create time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    splits = []
    for train_idx, val_idx in tscv.split(features_df):
        train_data = features_df.iloc[train_idx]
        val_data = features_df.iloc[val_idx]
        splits.append((train_data, val_data))
    
    return splits
```

### Class Imbalance Handling

**Strategy:** Use balanced class weights and SMOTE for training

```python
def handle_class_imbalance(X, y):
    """Handle class imbalance using SMOTE."""
    from imblearn.over_sampling import SMOTE
    
    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled
```

## Model Training Pipeline

### Complete Training Process

```python
def train_model(features_df, config):
    """Complete model training pipeline."""
    
    # 1. Time-based split
    train_data, val_data = create_time_based_split(
        features_df, 
        config['train_end_date'], 
        config['val_start_date']
    )
    
    # 2. Feature preparation
    feature_cols = [col for col in features_df.columns 
                   if col.startswith(('roll', 'avg_vel', 'vel_drop', 'avg_spin', 'spin_drop', 'rest'))]
    
    X_train = train_data[feature_cols]
    y_train = train_data['label_injury_within_21d']
    
    X_val = val_data[feature_cols]
    y_val = val_data['label_injury_within_21d']
    
    # 3. Handle missing values
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_train.median())
    
    # 4. Feature selection
    X_train_selected, selected_features, selector = select_features(X_train, y_train, k=10)
    X_val_selected = selector.transform(X_val)
    
    # 5. Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train_selected, y_train)
    
    # 6. Scale features
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train_balanced, X_val_selected)
    
    # 7. Train model
    model = InjuryRiskModel()
    model.fit(X_train_scaled, y_train_balanced)
    
    # 8. Evaluate on validation set
    val_predictions = model.predict_proba(X_val_scaled)[:, 1]
    metrics = evaluate_model(y_val, val_predictions)
    
    return model, metrics, selected_features
```

## Model Evaluation

### Primary Metrics

1. **AUC-ROC**: Overall model performance
2. **Precision@Top-10%**: Precision among highest risk predictions
3. **Recall@Top-10%**: Recall among highest risk predictions
4. **Brier Score**: Probability calibration quality
5. **Calibration Error**: How well probabilities match actual rates

### Evaluation Functions

```python
def evaluate_model(y_true, y_pred_proba):
    """Evaluate model performance."""
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss
    from sklearn.calibration import calibration_curve
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Precision/Recall at top 10%
    top_10_threshold = np.percentile(y_pred_proba, 90)
    top_10_mask = y_pred_proba >= top_10_threshold
    precision_top_10 = precision_score(y_true[top_10_mask], y_pred[top_10_mask], zero_division=0)
    recall_top_10 = recall_score(y_true[top_10_mask], y_pred[top_10_mask], zero_division=0)
    
    # Brier score
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # Calibration
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=10)
    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    
    return {
        'auc': auc,
        'precision_at_top_10': precision_top_10,
        'recall_at_top_10': recall_top_10,
        'brier_score': brier,
        'calibration_error': calibration_error,
        'positive_rate': y_true.mean(),
        'predicted_positive_rate': y_pred.mean()
    }
```

### Calibration Assessment

```python
def assess_calibration(y_true, y_pred_proba):
    """Assess probability calibration."""
    from sklearn.calibration import calibration_curve
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10
    )
    
    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    
    return fraction_of_positives, mean_predicted_value
```

## Feature Importance Analysis

### Coefficient Analysis

```python
def analyze_feature_importance(model, feature_names):
    """Analyze feature importance from logistic regression coefficients."""
    
    # Get coefficients
    coefficients = model.named_steps['calibratedclassifiercv'].base_estimator.coef_[0]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
        'direction': np.where(coefficients > 0, 'positive', 'negative')
    })
    
    # Sort by absolute importance
    importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
    
    return importance_df
```

### SHAP Analysis

```python
def generate_shap_explanations(model, X_sample, feature_names):
    """Generate SHAP explanations for model predictions."""
    import shap
    
    # Create SHAP explainer
    explainer = shap.LinearExplainer(model, X_sample)
    
    # Generate explanations
    shap_values = explainer.shap_values(X_sample)
    
    return shap_values, explainer
```

## Model Deployment

### Model Serialization

```python
def save_model(model, config, metrics, feature_names, model_path):
    """Save trained model and metadata."""
    import joblib
    
    # Create model artifact
    model_artifact = {
        'model': model,
        'config': config,
        'metrics': metrics,
        'feature_names': feature_names,
        'created_at': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    # Save to disk
    joblib.dump(model_artifact, model_path)
    
    return model_path
```

### Model Loading

```python
def load_model(model_path):
    """Load trained model from disk."""
    import joblib
    
    model_artifact = joblib.load(model_path)
    
    return model_artifact
```

## Prediction Pipeline

### Feature Generation for Prediction

```python
def generate_prediction_features(pitcher_id, as_of_date, appearances_df):
    """Generate features for a specific pitcher and date."""
    
    # Get pitcher's appearance data up to as_of_date
    pitcher_data = appearances_df[
        (appearances_df['pitcher_id'] == pitcher_id) &
        (appearances_df['game_date'] <= as_of_date)
    ].sort_values('game_date')
    
    if len(pitcher_data) == 0:
        return None
    
    # Generate features
    features = {}
    
    # Rolling workload features
    features['roll3g_pitch_count'] = pitcher_data['pitches_thrown'].tail(3).sum()
    features['roll7d_pitch_count'] = pitcher_data[
        pitcher_data['game_date'] >= as_of_date - pd.Timedelta(days=7)
    ]['pitches_thrown'].sum()
    
    # Velocity features
    recent_vel = pitcher_data[
        pitcher_data['game_date'] >= as_of_date - pd.Timedelta(days=7)
    ]['avg_vel']
    if len(recent_vel) > 0:
        features['avg_vel_7d'] = recent_vel.mean()
    
    # Rest days
    if len(pitcher_data) > 1:
        features['rest_days'] = (as_of_date - pitcher_data['game_date'].iloc[-1]).days
    else:
        features['rest_days'] = 0
    
    return features
```

### Risk Prediction

```python
def predict_risk(model, features, feature_names):
    """Predict injury risk for a pitcher."""
    
    # Prepare feature vector
    feature_vector = np.array([features.get(feature, 0) for feature in feature_names])
    
    # Make prediction
    risk_probability = model.predict_proba(feature_vector.reshape(1, -1))[0, 1]
    
    # Convert to percentage
    risk_percentage = risk_probability * 100
    
    # Determine risk level
    if risk_percentage < 20:
        risk_level = 'low'
    elif risk_percentage < 50:
        risk_level = 'medium'
    else:
        risk_level = 'high'
    
    return {
        'risk_score': risk_probability,
        'risk_percentage': risk_percentage,
        'risk_level': risk_level
    }
```

## Model Monitoring

### Performance Tracking

```python
def track_model_performance(model, new_data, config):
    """Track model performance on new data."""
    
    # Generate predictions
    predictions = model.predict_proba(new_data['features'])[:, 1]
    
    # Calculate metrics
    metrics = evaluate_model(new_data['labels'], predictions)
    
    # Store metrics
    store_performance_metrics(metrics, config['model_version'])
    
    # Alert if performance degrades
    if metrics['auc'] < config['min_auc_threshold']:
        alert_performance_degradation(metrics)
    
    return metrics
```

### Drift Detection

```python
def detect_data_drift(reference_data, current_data, features):
    """Detect data drift in features."""
    from scipy import stats
    
    drift_results = {}
    
    for feature in features:
        # KS test for distribution drift
        statistic, p_value = stats.ks_2samp(
            reference_data[feature], 
            current_data[feature]
        )
        
        drift_results[feature] = {
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': p_value < 0.05
        }
    
    return drift_results
```

## Success Criteria

### MVP 1 Success Metrics

1. **AUC > 0.75**: Model discriminates well between injury and non-injury cases
2. **Precision@Top-10% > 0.60**: High precision among highest risk predictions
3. **Recall@Top-10% > 0.70**: Captures most injuries in high-risk group
4. **Brier Score < 0.15**: Well-calibrated probability estimates
5. **Calibration Error < 0.05**: Probabilities match actual injury rates

### Model Validation

```python
def validate_model_performance(metrics, thresholds):
    """Validate model meets performance thresholds."""
    
    validation_results = {
        'auc_passed': metrics['auc'] > thresholds['min_auc'],
        'precision_passed': metrics['precision_at_top_10'] > thresholds['min_precision'],
        'recall_passed': metrics['recall_at_top_10'] > thresholds['min_recall'],
        'calibration_passed': metrics['calibration_error'] < thresholds['max_calibration_error']
    }
    
    all_passed = all(validation_results.values())
    
    return validation_results, all_passed
```

## Future Enhancements

### Phase 2 Improvements

1. **Ensemble Methods**: Combine multiple models for better performance
2. **Deep Learning**: Neural networks for complex pattern recognition
3. **Time Series Models**: LSTM/GRU for temporal dependencies
4. **Personalized Models**: Pitcher-specific model adjustments
5. **Multi-task Learning**: Predict injury type and severity

### Advanced Features

1. **Biomechanical Data**: Integration with motion capture data
2. **Wearable Sensors**: Real-time physiological monitoring
3. **Environmental Factors**: Weather, travel, schedule impact
4. **Team Context**: Bullpen usage, rotation position
5. **Historical Patterns**: Career injury history analysis
