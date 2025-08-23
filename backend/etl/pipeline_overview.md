# ETL Pipeline Overview

## Pipeline Architecture

The PitchGuard ETL pipeline transforms raw baseball data into ML-ready features through a series of stages designed for reliability, scalability, and data quality.

```
Raw Data Sources
    ↓
┌─────────────────┐
│   Ingestion     │ ← Baseball Savant API, Mock Data
│   Stage         │ ← Data validation & cleaning
└─────────────────┘
    ↓
┌─────────────────┐
│  Aggregation    │ ← Pitch → Appearance aggregation
│   Stage         │ ← Velocity/spin averaging
└─────────────────┘
    ↓
┌─────────────────┐
│  Feature        │ ← Rolling window calculations
│ Engineering     │ ← Trend analysis
│   Stage         │ ← Recovery patterns
└─────────────────┘
    ↓
┌─────────────────┐
│   Labeling      │ ← 21-day injury windows
│   Stage         │ ← Leakage prevention
└─────────────────┘
    ↓
Feature Store (Database)
```

## Pipeline Stages

### 1. Ingestion Stage

**Purpose**: Collect and validate raw data from multiple sources
**Input**: Baseball Savant API, mock data files, IL transaction data
**Output**: Cleaned, validated raw data

#### Data Sources

1. **Baseball Savant API**
   - Pitch-level data (release speed, spin rate, pitch type)
   - Game metadata (dates, teams, players)
   - Real-time updates during season

2. **Mock Data Generator**
   - Realistic pitch data for development/testing
   - Configurable patterns (velocity decline, high workload)
   - Injury scenarios for model validation

3. **IL Transaction Data**
   - Injury list transactions from MLB
   - Stint start/end dates
   - Injury type classifications

#### Validation Rules

```python
# Pitch data validation
def validate_pitch_data(df):
    """Validate pitch-level data quality."""
    # Velocity range check
    mask = (df['release_speed'] >= 50) & (df['release_speed'] <= 110)
    invalid_vel = df[~mask]
    
    # Spin rate range check
    mask = (df['release_spin_rate'] >= 0) & (df['release_spin_rate'] <= 3500)
    invalid_spin = df[~mask]
    
    # Date range check
    mask = df['game_date'] >= pd.to_datetime('2015-01-01')
    invalid_dates = df[~mask]
    
    return {
        'valid_data': df[mask],
        'invalid_velocity': invalid_vel,
        'invalid_spin': invalid_spin,
        'invalid_dates': invalid_dates
    }
```

#### Error Handling

- **Missing data**: Log and flag for review
- **Outlier detection**: Flag values outside expected ranges
- **Duplicate detection**: Remove duplicate records
- **Data type validation**: Ensure correct data types

### 2. Aggregation Stage

**Purpose**: Transform pitch-level data into game-level appearances
**Input**: Cleaned pitch data
**Output**: Aggregated appearance data

#### Aggregation Logic

```python
def aggregate_pitches_to_appearances(pitches_df):
    """Aggregate pitch data to game-level appearances."""
    appearances = pitches_df.groupby(['pitcher_id', 'game_date']).agg({
        'pitch_number': 'max',  # Total pitches thrown
        'release_speed': ['mean', 'std'],  # Velocity stats
        'release_spin_rate': ['mean', 'std'],  # Spin rate stats
        'at_bat_id': 'nunique'  # Number of at-bats
    }).reset_index()
    
    # Flatten column names
    appearances.columns = [
        'pitcher_id', 'game_date', 'pitches_thrown', 
        'avg_vel', 'vel_std', 'avg_spin', 'spin_std', 'at_bats'
    ]
    
    return appearances
```

#### Quality Checks

- **Minimum pitch threshold**: Require at least 1 pitch per appearance
- **Velocity consistency**: Flag large standard deviations
- **Missing data handling**: Handle NULL values appropriately

### 3. Feature Engineering Stage

**Purpose**: Compute rolling workload features and trend indicators
**Input**: Aggregated appearance data
**Output**: Feature snapshots for ML

#### Feature Categories

1. **Rolling Workload Features**
   ```python
   def compute_rolling_workloads(appearances_df):
       """Compute rolling pitch count features."""
       features = []
       
       for pitcher_id in appearances_df['pitcher_id'].unique():
           pitcher_data = appearances_df[
               appearances_df['pitcher_id'] == pitcher_id
           ].sort_values('game_date')
           
           # Rolling game counts (last 3 games)
           pitcher_data['roll3g_pitch_count'] = pitcher_data['pitches_thrown'].rolling(3).sum()
           
           # Rolling day counts (3, 7, 14 days)
           for days in [3, 7, 14]:
               col_name = f'roll{days}d_pitch_count'
               pitcher_data[col_name] = pitcher_data.set_index('game_date')[
                   'pitches_thrown'
               ].rolling(f'{days}D').sum().values
           
           features.append(pitcher_data)
       
       return pd.concat(features, ignore_index=True)
   ```

2. **Velocity/Spin Trend Features**
   ```python
   def compute_velocity_trends(appearances_df):
       """Compute velocity trend features."""
       features = []
       
       for pitcher_id in appearances_df['pitcher_id'].unique():
           pitcher_data = appearances_df[
               appearances_df['pitcher_id'] == pitcher_id
           ].sort_values('game_date')
           
           # 7-day average velocity
           pitcher_data['avg_vel_7d'] = pitcher_data.set_index('game_date')[
               'avg_vel'
           ].rolling('7D').mean().values
           
           # 30-day average velocity
           pitcher_data['avg_vel_30d'] = pitcher_data.set_index('game_date')[
               'avg_vel'
           ].rolling('30D').mean().values
           
           # Velocity drop vs 30-day baseline
           pitcher_data['vel_drop_vs_30d'] = (
               pitcher_data['avg_vel_7d'] - pitcher_data['avg_vel_30d']
           )
           
           features.append(pitcher_data)
       
       return pd.concat(features, ignore_index=True)
   ```

3. **Recovery Pattern Features**
   ```python
   def compute_recovery_features(appearances_df):
       """Compute rest and recovery features."""
       features = []
       
       for pitcher_id in appearances_df['pitcher_id'].unique():
           pitcher_data = appearances_df[
               appearances_df['pitcher_id'] == pitcher_id
           ].sort_values('game_date')
           
           # Days since last appearance
           pitcher_data['rest_days'] = pitcher_data['game_date'].diff().dt.days
           
           # Fill first appearance with 0
           pitcher_data['rest_days'] = pitcher_data['rest_days'].fillna(0)
           
           features.append(pitcher_data)
       
       return pd.concat(features, ignore_index=True)
   ```

#### Data Completeness Assessment

```python
def assess_data_completeness(features_df):
    """Assess data completeness for each feature snapshot."""
    def completeness_level(row):
        # Check 7-day pitch count
        if row['roll7d_pitch_count'] < 10:
            return 'low'
        
        # Check velocity data
        if pd.isna(row['avg_vel_7d']):
            return 'low'
        
        # Check 30-day baseline
        if pd.isna(row['vel_drop_vs_30d']):
            return 'med'
        
        return 'high'
    
    features_df['data_completeness'] = features_df.apply(completeness_level, axis=1)
    return features_df
```

### 4. Labeling Stage

**Purpose**: Generate injury labels for supervised learning
**Input**: Feature snapshots, injury data
**Output**: Labeled training data

#### Labeling Logic

```python
def generate_injury_labels(features_df, injuries_df, window_days=21):
    """Generate 21-day forward-looking injury labels."""
    labeled_features = []
    
    for pitcher_id in features_df['pitcher_id'].unique():
        pitcher_features = features_df[
            features_df['pitcher_id'] == pitcher_id
        ].copy()
        
        pitcher_injuries = injuries_df[
            injuries_df['pitcher_id'] == pitcher_id
        ].copy()
        
        # Generate labels for each feature snapshot
        for idx, row in pitcher_features.iterrows():
            as_of_date = row['as_of_date']
            
            # Look for injuries in 21-day window
            window_start = as_of_date + pd.Timedelta(days=1)
            window_end = as_of_date + pd.Timedelta(days=window_days)
            
            # Check if any injury starts in window
            injury_in_window = pitcher_injuries[
                (pitcher_injuries['il_start'] >= window_start) &
                (pitcher_injuries['il_start'] <= window_end)
            ]
            
            # Label: True if injury in window, False otherwise
            row['label_injury_within_21d'] = len(injury_in_window) > 0
            
            labeled_features.append(row)
    
    return pd.DataFrame(labeled_features)
```

#### Leakage Prevention

- **Forward-looking only**: Only use injury data available as of feature date
- **No future information**: Ensure no future data leaks into features
- **Temporal validation**: Validate time ordering in train/test splits

## Pipeline Orchestration

### Daily Pipeline

```python
class DailyETLPipeline:
    """Orchestrates daily ETL pipeline execution."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run(self, target_date):
        """Run complete ETL pipeline for target date."""
        try:
            # Stage 1: Ingestion
            self.logger.info(f"Starting ingestion for {target_date}")
            raw_data = self.ingestion_stage.run(target_date)
            
            # Stage 2: Aggregation
            self.logger.info("Starting aggregation")
            appearances = self.aggregation_stage.run(raw_data)
            
            # Stage 3: Feature Engineering
            self.logger.info("Starting feature engineering")
            features = self.feature_stage.run(appearances, target_date)
            
            # Stage 4: Labeling
            self.logger.info("Starting labeling")
            labeled_features = self.labeling_stage.run(features, target_date)
            
            # Store results
            self.store_results(labeled_features)
            
            self.logger.info("ETL pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {str(e)}")
            self.handle_failure(e)
            return False
```

### Incremental Processing

```python
def incremental_update(target_date):
    """Process only new data since last update."""
    # Get last processed date
    last_processed = get_last_processed_date()
    
    # Fetch new data only
    new_data = fetch_data_since(last_processed)
    
    # Process new data
    processed_data = process_incremental_data(new_data)
    
    # Update feature store
    update_feature_store(processed_data)
    
    # Update last processed date
    update_last_processed_date(target_date)
```

## Data Quality Monitoring

### Quality Metrics

1. **Completeness**: Percentage of non-null values
2. **Accuracy**: Data within expected ranges
3. **Consistency**: Data format and type consistency
4. **Timeliness**: Data freshness and processing latency

### Monitoring Dashboard

```python
def generate_quality_report(pipeline_run):
    """Generate data quality report for pipeline run."""
    report = {
        'run_id': pipeline_run.id,
        'timestamp': pipeline_run.timestamp,
        'metrics': {
            'records_processed': pipeline_run.records_processed,
            'records_failed': pipeline_run.records_failed,
            'completeness_rate': pipeline_run.completeness_rate,
            'processing_time': pipeline_run.processing_time
        },
        'alerts': pipeline_run.alerts,
        'recommendations': pipeline_run.recommendations
    }
    
    return report
```

## Error Handling & Recovery

### Error Types

1. **Data Source Errors**: API failures, network issues
2. **Validation Errors**: Data quality issues
3. **Processing Errors**: Computation failures
4. **Storage Errors**: Database connection issues

### Recovery Strategies

```python
def handle_pipeline_failure(error, stage, context):
    """Handle pipeline failures with appropriate recovery."""
    if isinstance(error, DataSourceError):
        # Retry with exponential backoff
        return retry_with_backoff(context, max_retries=3)
    
    elif isinstance(error, ValidationError):
        # Log and continue with partial data
        log_validation_error(error)
        return continue_with_partial_data(context)
    
    elif isinstance(error, ProcessingError):
        # Restart from last successful stage
        return restart_from_stage(context, stage)
    
    else:
        # Unknown error - alert and stop
        alert_unknown_error(error)
        return False
```

## Performance Optimization

### Parallel Processing

```python
def parallel_feature_computation(features_df, num_workers=4):
    """Compute features in parallel for multiple pitchers."""
    from concurrent.futures import ProcessPoolExecutor
    
    # Split data by pitcher
    pitcher_groups = [group for _, group in features_df.groupby('pitcher_id')]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_pitcher_features, pitcher_groups))
    
    return pd.concat(results, ignore_index=True)
```

### Caching Strategy

```python
def cache_intermediate_results(stage_output, cache_key):
    """Cache intermediate results for reuse."""
    cache_path = f"/tmp/pitchguard/cache/{cache_key}.parquet"
    
    # Save to cache
    stage_output.to_parquet(cache_path)
    
    return cache_path

def load_cached_results(cache_key):
    """Load cached results if available."""
    cache_path = f"/tmp/pitchguard/cache/{cache_key}.parquet"
    
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    
    return None
```

## Testing Strategy

### Unit Tests

```python
def test_feature_computation():
    """Test feature computation logic."""
    # Create test data
    test_appearances = create_test_appearances()
    
    # Compute features
    features = compute_rolling_workloads(test_appearances)
    
    # Assert expected results
    assert 'roll7d_pitch_count' in features.columns
    assert features['roll7d_pitch_count'].min() >= 0
    assert not features['roll7d_pitch_count'].isna().any()
```

### Integration Tests

```python
def test_end_to_end_pipeline():
    """Test complete pipeline with mock data."""
    # Setup test environment
    test_config = create_test_config()
    pipeline = DailyETLPipeline(test_config)
    
    # Run pipeline
    success = pipeline.run(test_date)
    
    # Verify results
    assert success
    assert verify_feature_store_updated(test_date)
```

### Performance Tests

```python
def test_pipeline_performance():
    """Test pipeline performance with large datasets."""
    # Create large test dataset
    large_dataset = create_large_test_dataset()
    
    # Measure processing time
    start_time = time.time()
    pipeline.run(large_dataset)
    processing_time = time.time() - start_time
    
    # Assert performance requirements
    assert processing_time < 300  # 5 minutes max
```
