# QA Test Plan

## Overview

This document outlines the comprehensive testing strategy for PitchGuard, covering data validation, model testing, API testing, and UI testing to ensure system reliability and accuracy.

## Testing Strategy

### Test Pyramid

```
┌─────────────────────────────────────────────────────────┐
│                    E2E Tests (10%)                      │
│                 (Complete workflows)                    │
├─────────────────────────────────────────────────────────┤
│                  Integration Tests (20%)                │
│                (API endpoints, DB)                     │
├─────────────────────────────────────────────────────────┤
│                   Unit Tests (70%)                      │
│              (Individual components)                    │
└─────────────────────────────────────────────────────────┘
```

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Component interaction testing
3. **End-to-End Tests**: Complete workflow testing
4. **Performance Tests**: Load and stress testing
5. **Data Quality Tests**: Data validation and integrity

## Data Testing

### 1. Data Validation Tests

**Purpose**: Ensure data quality and integrity throughout the pipeline

#### Test Cases

```python
def test_pitch_data_validation():
    """Test pitch data validation rules."""
    
    # Test velocity range validation
    valid_velocities = [50.0, 95.5, 110.0]
    invalid_velocities = [49.9, 110.1, -5.0]
    
    for vel in valid_velocities:
        assert is_valid_velocity(vel), f"Velocity {vel} should be valid"
    
    for vel in invalid_velocities:
        assert not is_valid_velocity(vel), f"Velocity {vel} should be invalid"

def test_spin_rate_validation():
    """Test spin rate validation rules."""
    
    # Test spin rate range validation
    valid_spin_rates = [0, 2000, 3500]
    invalid_spin_rates = [-1, 3501, 5000]
    
    for spin in valid_spin_rates:
        assert is_valid_spin_rate(spin), f"Spin rate {spin} should be valid"
    
    for spin in invalid_spin_rates:
        assert not is_valid_spin_rate(spin), f"Spin rate {spin} should be invalid"

def test_date_validation():
    """Test date validation rules."""
    
    # Test date range validation
    valid_dates = ['2015-01-01', '2024-04-15', '2024-12-31']
    invalid_dates = ['2014-12-31', '2025-01-01']
    
    for date in valid_dates:
        assert is_valid_game_date(date), f"Date {date} should be valid"
    
    for date in invalid_dates:
        assert not is_valid_game_date(date), f"Date {date} should be invalid"
```

#### Data Completeness Tests

```python
def test_data_completeness():
    """Test data completeness requirements."""
    
    # Test minimum data requirements
    test_data = create_test_dataset()
    
    # Check required fields
    required_fields = ['game_date', 'pitcher_id', 'pitch_type', 'pitch_number']
    for field in required_fields:
        assert field in test_data.columns, f"Required field {field} missing"
    
    # Check for missing values in critical fields
    critical_fields = ['pitcher_id', 'game_date']
    for field in critical_fields:
        missing_count = test_data[field].isnull().sum()
        assert missing_count == 0, f"Critical field {field} has {missing_count} missing values"

def test_data_consistency():
    """Test data consistency across tables."""
    
    # Test referential integrity
    pitches_df = load_pitches_data()
    appearances_df = load_appearances_data()
    
    # All pitcher_ids in pitches should exist in appearances
    pitch_pitchers = set(pitches_df['pitcher_id'].unique())
    appearance_pitchers = set(appearances_df['pitcher_id'].unique())
    
    assert pitch_pitchers.issubset(appearance_pitchers), \
        "All pitchers in pitches table should exist in appearances table"
```

### 2. Feature Engineering Tests

**Purpose**: Validate feature computation accuracy

#### Rolling Window Tests

```python
def test_rolling_window_calculation():
    """Test rolling window feature calculations."""
    
    # Create test data
    test_appearances = pd.DataFrame({
        'pitcher_id': [1, 1, 1, 1, 1],
        'game_date': ['2024-04-01', '2024-04-05', '2024-04-10', '2024-04-15', '2024-04-20'],
        'pitches_thrown': [85, 95, 90, 100, 88]
    })
    
    # Test 3-game rolling sum
    features = compute_rolling_features(test_appearances)
    
    # Expected: Last 3 games = 90 + 100 + 88 = 278
    expected_3g_sum = 278
    actual_3g_sum = features.iloc[-1]['roll3g_pitch_count']
    
    assert actual_3g_sum == expected_3g_sum, \
        f"Expected 3-game sum {expected_3g_sum}, got {actual_3g_sum}"

def test_velocity_trend_calculation():
    """Test velocity trend feature calculations."""
    
    # Create test data with known velocity patterns
    test_appearances = pd.DataFrame({
        'pitcher_id': [1] * 10,
        'game_date': pd.date_range('2024-04-01', periods=10, freq='D'),
        'avg_vel': [95.0, 94.5, 94.0, 93.5, 93.0, 92.5, 92.0, 91.5, 91.0, 90.5]
    })
    
    # Test velocity drop calculation
    features = compute_velocity_features(test_appearances)
    
    # Expected: 7-day avg (last 7 values) - 30-day avg (all 10 values)
    expected_7d_avg = 91.5  # (90.5 + 91.0 + 91.5 + 92.0 + 92.5 + 93.0 + 93.5) / 7
    expected_30d_avg = 92.75  # (95.0 + 94.5 + ... + 90.5) / 10
    expected_drop = expected_7d_avg - expected_30d_avg
    
    actual_drop = features.iloc[-1]['vel_drop_vs_30d']
    
    assert abs(actual_drop - expected_drop) < 0.01, \
        f"Expected velocity drop {expected_drop}, got {actual_drop}"
```

### 3. Label Generation Tests

**Purpose**: Validate injury label generation logic

```python
def test_injury_label_generation():
    """Test injury label generation with known scenarios."""
    
    # Create test data with known injury
    test_features = pd.DataFrame({
        'pitcher_id': [1] * 30,
        'as_of_date': pd.date_range('2024-04-01', periods=30, freq='D'),
        'roll7d_pitch_count': [100] * 30
    })
    
    test_injuries = pd.DataFrame({
        'pitcher_id': [1],
        'il_start': ['2024-04-15'],  # Injury starts on April 15
        'il_end': ['2024-05-01']
    })
    
    # Generate labels
    labeled_features = generate_injury_labels(test_features, test_injuries, window_days=21)
    
    # Test specific dates
    # April 10: Should be labeled True (injury within 21 days)
    april_10_label = labeled_features[
        labeled_features['as_of_date'] == '2024-04-10'
    ]['label_injury_within_21d'].iloc[0]
    assert april_10_label == True, "April 10 should be labeled as injury risk"
    
    # April 5: Should be labeled True (injury within 21 days)
    april_5_label = labeled_features[
        labeled_features['as_of_date'] == '2024-04-05'
    ]['label_injury_within_21d'].iloc[0]
    assert april_5_label == True, "April 5 should be labeled as injury risk"
    
    # March 25: Should be labeled False (injury not within 21 days)
    march_25_label = labeled_features[
        labeled_features['as_of_date'] == '2024-03-25'
    ]['label_injury_within_21d'].iloc[0]
    assert march_25_label == False, "March 25 should not be labeled as injury risk"

def test_no_future_leakage():
    """Test that no future information leaks into labels."""
    
    # Create test data
    test_features = pd.DataFrame({
        'pitcher_id': [1] * 10,
        'as_of_date': pd.date_range('2024-04-01', periods=10, freq='D'),
        'roll7d_pitch_count': [100] * 10
    })
    
    # Injury happens on April 15
    test_injuries = pd.DataFrame({
        'pitcher_id': [1],
        'il_start': ['2024-04-15'],
        'il_end': ['2024-05-01']
    })
    
    # Generate labels
    labeled_features = generate_injury_labels(test_features, test_injuries, window_days=21)
    
    # Check that April 15 and later dates don't have access to future injury info
    for date in ['2024-04-15', '2024-04-16', '2024-04-17']:
        label = labeled_features[
            labeled_features['as_of_date'] == date
        ]['label_injury_within_21d'].iloc[0]
        # These should be False because injury info isn't available yet
        assert label == False, f"Date {date} should not have future injury info"
```

## Model Testing

### 1. Model Training Tests

**Purpose**: Validate model training process

```python
def test_model_training_pipeline():
    """Test complete model training pipeline."""
    
    # Create test data
    test_features = create_test_features_dataset()
    
    # Train model
    model, metrics, feature_names = train_model(test_features, test_config)
    
    # Test model object
    assert hasattr(model, 'predict_proba'), "Model should have predict_proba method"
    assert hasattr(model, 'predict'), "Model should have predict method"
    
    # Test metrics
    assert 'auc' in metrics, "Metrics should include AUC"
    assert 'precision_at_top_10' in metrics, "Metrics should include precision@top10"
    assert 'recall_at_top_10' in metrics, "Metrics should include recall@top10"
    
    # Test performance thresholds
    assert metrics['auc'] > 0.7, f"AUC {metrics['auc']} should be > 0.7"
    assert metrics['precision_at_top_10'] > 0.5, f"Precision {metrics['precision_at_top_10']} should be > 0.5"

def test_time_based_split():
    """Test time-based train/validation split."""
    
    # Create test data with dates
    test_features = pd.DataFrame({
        'as_of_date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'pitcher_id': [1] * 100,
        'label_injury_within_21d': [False] * 95 + [True] * 5
    })
    
    # Create split
    train_data, val_data = create_time_based_split(
        test_features, 
        train_end_date='2024-03-01',
        val_start_date='2024-03-02'
    )
    
    # Test no overlap
    assert train_data['as_of_date'].max() < val_data['as_of_date'].min(), \
        "Training and validation sets should not overlap"
    
    # Test data distribution
    assert len(train_data) > 0, "Training set should not be empty"
    assert len(val_data) > 0, "Validation set should not be empty"

def test_feature_selection():
    """Test feature selection process."""
    
    # Create test data with known feature importance
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 2, 100)
    
    # Add one highly predictive feature
    X[:, 0] = y * 2 + np.random.randn(100) * 0.1
    
    # Test feature selection
    X_selected, selected_features, selector = select_features(X, y, k=5)
    
    # Test output shape
    assert X_selected.shape[1] == 5, f"Expected 5 features, got {X_selected.shape[1]}"
    assert len(selected_features) == 5, f"Expected 5 feature names, got {len(selected_features)}"
    
    # Test that most predictive feature is selected
    assert 0 in selected_features, "Most predictive feature should be selected"
```

### 2. Model Prediction Tests

**Purpose**: Validate model prediction accuracy

```python
def test_model_prediction_consistency():
    """Test that model predictions are consistent."""
    
    # Load trained model
    model = load_model('./models/test_model.pkl')
    
    # Create test features
    test_features = np.random.randn(10, 10)
    
    # Make predictions
    predictions_1 = model.predict_proba(test_features)
    predictions_2 = model.predict_proba(test_features)
    
    # Test consistency
    np.testing.assert_array_almost_equal(predictions_1, predictions_2, decimal=10), \
        "Model predictions should be consistent"

def test_prediction_probability_range():
    """Test that predictions are valid probabilities."""
    
    # Load trained model
    model = load_model('./models/test_model.pkl')
    
    # Create test features
    test_features = np.random.randn(100, 10)
    
    # Make predictions
    predictions = model.predict_proba(test_features)[:, 1]
    
    # Test probability range
    assert np.all(predictions >= 0), "All predictions should be >= 0"
    assert np.all(predictions <= 1), "All predictions should be <= 1"

def test_feature_importance_consistency():
    """Test feature importance analysis."""
    
    # Load trained model
    model = load_model('./models/test_model.pkl')
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(model, feature_names)
    
    # Test output structure
    assert 'feature' in importance_df.columns, "Importance df should have 'feature' column"
    assert 'coefficient' in importance_df.columns, "Importance df should have 'coefficient' column"
    assert 'direction' in importance_df.columns, "Importance df should have 'direction' column"
    
    # Test all features included
    assert len(importance_df) == len(feature_names), "All features should be included"
    
    # Test directions are valid
    valid_directions = ['positive', 'negative']
    assert all(direction in valid_directions for direction in importance_df['direction']), \
        "All directions should be 'positive' or 'negative'"
```

## API Testing

### 1. API Endpoint Tests

**Purpose**: Validate API endpoint functionality

```python
def test_risk_assessment_endpoint():
    """Test risk assessment API endpoint."""
    
    # Test valid request
    response = client.post("/api/v1/risk/pitcher", json={
        "pitcher_id": 12345,
        "as_of_date": "2024-06-15"
    })
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert data['success'] == True, "Response should indicate success"
    assert 'risk_score' in data['data'], "Response should include risk_score"
    assert 'risk_level' in data['data'], "Response should include risk_level"
    assert 'top_contributors' in data['data'], "Response should include top_contributors"

def test_workload_history_endpoint():
    """Test workload history API endpoint."""
    
    # Test valid request
    response = client.get("/api/v1/workload/pitcher/12345?start_date=2024-06-01&end_date=2024-06-15")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert data['success'] == True, "Response should indicate success"
    assert 'workload_series' in data['data'], "Response should include workload_series"
    assert 'summary_stats' in data['data'], "Response should include summary_stats"

def test_pitchers_list_endpoint():
    """Test pitchers list API endpoint."""
    
    # Test valid request
    response = client.get("/api/v1/pitchers?team=NYY&limit=5")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert data['success'] == True, "Response should indicate success"
    assert 'pitchers' in data['data'], "Response should include pitchers list"
    assert 'pagination' in data['data'], "Response should include pagination info"

def test_error_handling():
    """Test API error handling."""
    
    # Test invalid pitcher ID
    response = client.post("/api/v1/risk/pitcher", json={
        "pitcher_id": 99999,
        "as_of_date": "2024-06-15"
    })
    
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
    
    data = response.json()
    assert data['success'] == False, "Response should indicate failure"
    assert data['error']['code'] == 'UNKNOWN_PITCHER', "Should return UNKNOWN_PITCHER error"
    
    # Test invalid date
    response = client.post("/api/v1/risk/pitcher", json={
        "pitcher_id": 12345,
        "as_of_date": "2025-01-01"
    })
    
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    
    data = response.json()
    assert data['error']['code'] == 'INVALID_DATE', "Should return INVALID_DATE error"
```

### 2. API Performance Tests

**Purpose**: Validate API performance under load

```python
def test_api_response_time():
    """Test API response time requirements."""
    
    import time
    
    # Test single request response time
    start_time = time.time()
    response = client.post("/api/v1/risk/pitcher", json={
        "pitcher_id": 12345,
        "as_of_date": "2024-06-15"
    })
    end_time = time.time()
    
    response_time = end_time - start_time
    assert response_time < 2.0, f"Response time {response_time}s should be < 2.0s"
    assert response.status_code == 200, "Request should succeed"

def test_api_concurrent_requests():
    """Test API performance under concurrent load."""
    
    import concurrent.futures
    import time
    
    def make_request():
        return client.post("/api/v1/risk/pitcher", json={
            "pitcher_id": 12345,
            "as_of_date": "2024-06-15"
        })
    
    # Make 10 concurrent requests
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [future.result() for future in futures]
    end_time = time.time()
    
    total_time = end_time - start_time
    
    # All requests should succeed
    success_count = sum(1 for r in responses if r.status_code == 200)
    assert success_count == 10, f"All 10 requests should succeed, got {success_count}"
    
    # Total time should be reasonable
    assert total_time < 5.0, f"Total time {total_time}s should be < 5.0s"

def test_api_rate_limiting():
    """Test API rate limiting functionality."""
    
    # Make many requests quickly
    responses = []
    for _ in range(150):  # Exceed rate limit
        response = client.post("/api/v1/risk/pitcher", json={
            "pitcher_id": 12345,
            "as_of_date": "2024-06-15"
        })
        responses.append(response)
    
    # Some requests should be rate limited
    rate_limited_count = sum(1 for r in responses if r.status_code == 429)
    assert rate_limited_count > 0, "Some requests should be rate limited"
```

## UI Testing

### 1. Component Tests

**Purpose**: Validate React component functionality

```typescript
// RiskBadge component test
describe('RiskBadge', () => {
  it('should render correct risk level', () => {
    const { getByText } = render(<RiskBadge level="high" score={75} />);
    expect(getByText('High Risk')).toBeInTheDocument();
    expect(getByText('75%')).toBeInTheDocument();
  });

  it('should apply correct styling for risk levels', () => {
    const { container } = render(<RiskBadge level="medium" score={35} />);
    const badge = container.firstChild;
    expect(badge).toHaveClass('risk-badge--medium');
  });
});

// PitchersTable component test
describe('PitchersTable', () => {
  it('should render pitcher data correctly', () => {
    const mockPitchers = [
      { id: 1, name: 'Gerrit Cole', team: 'NYY', riskLevel: 'low', riskScore: 15 }
    ];
    
    const { getByText } = render(<PitchersTable pitchers={mockPitchers} />);
    expect(getByText('Gerrit Cole')).toBeInTheDocument();
    expect(getByText('NYY')).toBeInTheDocument();
  });

  it('should handle sorting', () => {
    const mockPitchers = [
      { id: 1, name: 'Pitcher A', riskScore: 25 },
      { id: 2, name: 'Pitcher B', riskScore: 15 }
    ];
    
    const { getByText } = render(<PitchersTable pitchers={mockPitchers} />);
    const riskHeader = getByText('Risk Score');
    
    fireEvent.click(riskHeader);
    
    // Check that sorting occurred
    const rows = screen.getAllByRole('row');
    expect(rows[1]).toHaveTextContent('Pitcher B'); // Lower risk first
    expect(rows[2]).toHaveTextContent('Pitcher A'); // Higher risk second
  });
});
```

### 2. Integration Tests

**Purpose**: Validate component interactions

```typescript
// Pitcher detail page integration test
describe('PitcherDetailPage', () => {
  it('should load and display pitcher data', async () => {
    // Mock API response
    server.use(
      rest.get('/api/v1/risk/pitcher/:id', (req, res, ctx) => {
        return res(ctx.json({
          success: true,
          data: {
            pitcher_id: 12345,
            risk_score: 0.23,
            risk_level: 'medium',
            top_contributors: []
          }
        }));
      })
    );

    render(<PitcherDetailPage pitcherId={12345} />);
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('Medium Risk')).toBeInTheDocument();
    });
    
    expect(screen.getByText('23%')).toBeInTheDocument();
  });

  it('should handle API errors gracefully', async () => {
    // Mock API error
    server.use(
      rest.get('/api/v1/risk/pitcher/:id', (req, res, ctx) => {
        return res(ctx.status(404), ctx.json({
          success: false,
          error: { code: 'UNKNOWN_PITCHER', message: 'Pitcher not found' }
        }));
      })
    );

    render(<PitcherDetailPage pitcherId={99999} />);
    
    await waitFor(() => {
      expect(screen.getByText('Pitcher not found')).toBeInTheDocument();
    });
  });
});
```

### 3. End-to-End Tests

**Purpose**: Validate complete user workflows

```typescript
// Complete workflow test
describe('Pitcher Risk Assessment Workflow', () => {
  it('should complete full risk assessment workflow', async () => {
    // Start at overview page
    render(<App />);
    
    // Navigate to pitcher detail
    const pitcherRow = screen.getByText('Gerrit Cole');
    fireEvent.click(pitcherRow);
    
    // Verify detail page loaded
    await waitFor(() => {
      expect(screen.getByText('Gerrit Cole')).toBeInTheDocument();
    });
    
    // Check risk assessment displayed
    expect(screen.getByText('Risk Assessment')).toBeInTheDocument();
    
    // Navigate back to overview
    const backButton = screen.getByText('Back to Overview');
    fireEvent.click(backButton);
    
    // Verify returned to overview
    expect(screen.getByText('Pitcher Overview')).toBeInTheDocument();
  });
});
```

## Performance Testing

### 1. Load Testing

**Purpose**: Validate system performance under load

```python
def test_system_load_capacity():
    """Test system capacity under load."""
    
    import asyncio
    import aiohttp
    
    async def make_request(session, request_id):
        async with session.post('http://localhost:8000/api/v1/risk/pitcher', 
                               json={"pitcher_id": 12345, "as_of_date": "2024-06-15"}) as response:
            return await response.json()
    
    async def load_test():
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session, i) for i in range(100)]
            results = await asyncio.gather(*tasks)
            return results
    
    # Run load test
    results = asyncio.run(load_test())
    
    # Analyze results
    success_count = sum(1 for r in results if r.get('success', False))
    assert success_count >= 95, f"At least 95% of requests should succeed, got {success_count}%"
```

### 2. Stress Testing

**Purpose**: Validate system behavior under stress

```python
def test_system_stress_limits():
    """Test system stress limits."""
    
    # Gradually increase load until system breaks
    load_levels = [10, 50, 100, 200, 500]
    
    for load in load_levels:
        try:
            # Make concurrent requests
            results = make_concurrent_requests(load)
            success_rate = sum(1 for r in results if r.status_code == 200) / load
            
            if success_rate < 0.9:
                print(f"System stress limit reached at {load} concurrent requests")
                break
                
        except Exception as e:
            print(f"System failed at {load} concurrent requests: {e}")
            break
```

## Test Data Management

### 1. Test Data Generation

```python
def create_test_dataset():
    """Create comprehensive test dataset."""
    
    # Generate realistic test data
    test_data = {
        'pitches': generate_test_pitches(1000),
        'appearances': generate_test_appearances(100),
        'injuries': generate_test_injuries(10)
    }
    
    return test_data

def generate_test_pitches(num_pitches):
    """Generate realistic pitch data."""
    
    pitches = []
    for i in range(num_pitches):
        pitch = {
            'game_date': f'2024-04-{(i % 30) + 1:02d}',
            'pitcher_id': (i % 5) + 1,
            'pitch_type': ['FF', 'SL', 'CH', 'CT'][i % 4],
            'release_speed': 90 + (i % 10),
            'release_spin_rate': 2000 + (i % 1000),
            'pitch_number': (i % 20) + 1,
            'at_bat_id': i
        }
        pitches.append(pitch)
    
    return pd.DataFrame(pitches)
```

### 2. Test Data Validation

```python
def validate_test_data(test_data):
    """Validate test data quality."""
    
    # Check data completeness
    assert len(test_data['pitches']) > 0, "Test pitches data should not be empty"
    assert len(test_data['appearances']) > 0, "Test appearances data should not be empty"
    
    # Check data quality
    assert test_data['pitches']['release_speed'].between(50, 110).all(), \
        "All pitch velocities should be in valid range"
    
    # Check referential integrity
    pitch_pitchers = set(test_data['pitches']['pitcher_id'].unique())
    appearance_pitchers = set(test_data['appearances']['pitcher_id'].unique())
    assert pitch_pitchers.issubset(appearance_pitchers), \
        "All pitchers in pitches should exist in appearances"
```

## Test Execution

### 1. Test Suite Organization

```
tests/
├── unit/
│   ├── test_data_validation.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   └── test_api_endpoints.py
├── integration/
│   ├── test_etl_pipeline.py
│   ├── test_model_prediction.py
│   └── test_api_integration.py
├── e2e/
│   ├── test_complete_workflow.py
│   └── test_user_scenarios.py
└── performance/
    ├── test_load_capacity.py
    └── test_stress_limits.py
```

### 2. Test Execution Commands

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
pytest tests/performance/

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# Run specific test file
pytest tests/unit/test_data_validation.py

# Run specific test function
pytest tests/unit/test_data_validation.py::test_pitch_data_validation
```

### 3. Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r backend/requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=backend --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Test Reporting

### 1. Test Results Summary

```python
def generate_test_report():
    """Generate comprehensive test report."""
    
    report = {
        'summary': {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0
        },
        'categories': {
            'unit': {'total': 0, 'passed': 0, 'failed': 0},
            'integration': {'total': 0, 'passed': 0, 'failed': 0},
            'e2e': {'total': 0, 'passed': 0, 'failed': 0},
            'performance': {'total': 0, 'passed': 0, 'failed': 0}
        },
        'coverage': {
            'backend': 0.0,
            'frontend': 0.0
        },
        'performance_metrics': {
            'api_response_time': 0.0,
            'load_capacity': 0,
            'stress_limit': 0
        }
    }
    
    return report
```

### 2. Quality Gates

```python
def check_quality_gates():
    """Check if quality gates are met."""
    
    gates = {
        'test_coverage': 80.0,  # Minimum 80% test coverage
        'api_response_time': 2.0,  # Maximum 2 second response time
        'load_capacity': 100,  # Minimum 100 concurrent requests
        'test_pass_rate': 95.0  # Minimum 95% test pass rate
    }
    
    results = {
        'test_coverage': get_test_coverage(),
        'api_response_time': get_api_response_time(),
        'load_capacity': get_load_capacity(),
        'test_pass_rate': get_test_pass_rate()
    }
    
    all_passed = True
    for gate, threshold in gates.items():
        if results[gate] < threshold:
            print(f"Quality gate failed: {gate} = {results[gate]} (threshold: {threshold})")
            all_passed = False
    
    return all_passed
```
