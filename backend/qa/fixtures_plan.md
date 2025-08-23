# Mock Data & Fixtures Plan

## Overview

This document outlines the strategy for creating realistic mock data and test fixtures for PitchGuard development, testing, and demonstration purposes.

## Mock Data Strategy

### Design Principles

1. **Realistic Patterns**: Data should follow real MLB patterns and distributions
2. **Reproducible**: Mock data should be deterministic and reproducible
3. **Comprehensive**: Cover all edge cases and scenarios
4. **Scalable**: Easy to generate different volumes of data
5. **Educational**: Demonstrate key risk patterns and injury scenarios

### Data Sources for Realism

- **MLB Statistics**: Use real MLB averages and distributions
- **Baseball Savant**: Reference actual pitch data patterns
- **Injury Research**: Base injury patterns on published studies
- **Expert Knowledge**: Incorporate baseball domain expertise

## Mock Data Generator Architecture

### Core Components

```python
class MockDataGenerator:
    """Generate realistic mock data for PitchGuard."""
    
    def __init__(self, config):
        self.config = config
        self.random_seed = config.get('random_seed', 42)
        self.pitcher_profiles = self._load_pitcher_profiles()
        self.injury_patterns = self._load_injury_patterns()
    
    def generate_complete_dataset(self, num_pitchers=3, start_date='2024-04-01', 
                                end_date='2024-06-30', injury_scenario=True):
        """Generate complete mock dataset."""
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Generate pitcher profiles
        pitchers = self._generate_pitcher_profiles(num_pitchers)
        
        # Generate pitch data
        pitches = self._generate_pitch_data(pitchers, start_date, end_date)
        
        # Generate appearances
        appearances = self._generate_appearances(pitches)
        
        # Generate injuries (if requested)
        injuries = []
        if injury_scenario:
            injuries = self._generate_injury_scenario(pitchers, appearances)
        
        return {
            'pitchers': pitchers,
            'pitches': pitches,
            'appearances': appearances,
            'injuries': injuries
        }
```

### Pitcher Profile Generation

```python
def _generate_pitcher_profiles(self, num_pitchers):
    """Generate realistic pitcher profiles."""
    
    pitcher_names = [
        "Gerrit Cole", "Carlos Rodón", "Nestor Cortes",
        "Luis Severino", "Domingo Germán", "Clarke Schmidt"
    ]
    
    teams = ["NYY", "LAD", "HOU", "ATL", "BOS", "CHC"]
    
    pitchers = []
    for i in range(num_pitchers):
        pitcher = {
            'pitcher_id': 10000 + i,
            'name': pitcher_names[i],
            'team': teams[i],
            'role': np.random.choice(['starter', 'reliever'], p=[0.7, 0.3]),
            'baseline_velocity': np.random.normal(94.5, 2.0),  # 92-97 MPH
            'baseline_spin_rate': np.random.normal(2400, 200),  # 2200-2600 RPM
            'injury_risk_profile': np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
        }
        pitchers.append(pitcher)
    
    return pitchers

def _load_pitcher_profiles(self):
    """Load realistic pitcher baseline profiles."""
    
    return {
        'velocity_baselines': {
            'starter': {'mean': 94.5, 'std': 2.0, 'min': 92, 'max': 97},
            'reliever': {'mean': 95.5, 'std': 2.5, 'min': 93, 'max': 98}
        },
        'spin_rate_baselines': {
            'starter': {'mean': 2400, 'std': 200, 'min': 2200, 'max': 2600},
            'reliever': {'mean': 2450, 'std': 250, 'min': 2200, 'max': 2700}
        },
        'pitch_count_patterns': {
            'starter': {'mean': 95, 'std': 15, 'min': 70, 'max': 120},
            'reliever': {'mean': 25, 'std': 10, 'min': 10, 'max': 50}
        }
    }
```

### Pitch Data Generation

```python
def _generate_pitch_data(self, pitchers, start_date, end_date):
    """Generate realistic pitch-level data."""
    
    pitches = []
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    for pitcher in pitchers:
        # Generate appearance schedule
        appearance_dates = self._generate_appearance_schedule(
            pitcher, start_dt, end_dt
        )
        
        for game_date in appearance_dates:
            # Generate pitches for this appearance
            game_pitches = self._generate_game_pitches(pitcher, game_date)
            pitches.extend(game_pitches)
    
    return pd.DataFrame(pitches)

def _generate_appearance_schedule(self, pitcher, start_date, end_date):
    """Generate realistic appearance schedule."""
    
    if pitcher['role'] == 'starter':
        # Starters: every 4-5 days
        interval_days = np.random.choice([4, 5], p=[0.6, 0.4])
    else:
        # Relievers: more frequent, irregular intervals
        interval_days = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
    
    appearance_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        # Add some randomness to interval
        actual_interval = interval_days + np.random.randint(-1, 2)
        current_date += pd.Timedelta(days=max(1, actual_interval))
        
        if current_date <= end_date:
            appearance_dates.append(current_date)
    
    return appearance_dates

def _generate_game_pitches(self, pitcher, game_date):
    """Generate pitches for a single game appearance."""
    
    # Determine number of pitches based on role and random variation
    if pitcher['role'] == 'starter':
        pitch_count = int(np.random.normal(95, 15))
        pitch_count = max(70, min(120, pitch_count))  # Clamp to realistic range
    else:
        pitch_count = int(np.random.normal(25, 10))
        pitch_count = max(10, min(50, pitch_count))
    
    # Generate individual pitches
    pitches = []
    for pitch_num in range(1, pitch_count + 1):
        pitch = self._generate_single_pitch(pitcher, game_date, pitch_num)
        pitches.append(pitch)
    
    return pitches

def _generate_single_pitch(self, pitcher, game_date, pitch_num):
    """Generate a single pitch with realistic characteristics."""
    
    # Pitch types with realistic frequencies
    pitch_types = ['FF', 'SL', 'CH', 'CT', 'CB', 'SI']
    pitch_weights = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]  # Four-seam most common
    
    pitch_type = np.random.choice(pitch_types, p=pitch_weights)
    
    # Velocity with realistic variation
    baseline_vel = pitcher['baseline_velocity']
    velocity_variation = np.random.normal(0, 1.5)  # ±3 MPH typical variation
    release_speed = baseline_vel + velocity_variation
    release_speed = max(85, min(105, release_speed))  # Realistic range
    
    # Spin rate correlated with velocity and pitch type
    baseline_spin = pitcher['baseline_spin_rate']
    if pitch_type == 'FF':
        spin_multiplier = 1.0
    elif pitch_type == 'SL':
        spin_multiplier = 1.1
    elif pitch_type == 'CH':
        spin_multiplier = 0.9
    else:
        spin_multiplier = 1.05
    
    spin_variation = np.random.normal(0, 100)
    release_spin_rate = int(baseline_spin * spin_multiplier + spin_variation)
    release_spin_rate = max(1800, min(3200, release_spin_rate))
    
    return {
        'game_date': game_date.strftime('%Y-%m-%d'),
        'pitcher_id': pitcher['pitcher_id'],
        'pitch_type': pitch_type,
        'release_speed': round(release_speed, 1),
        'release_spin_rate': release_spin_rate,
        'pitch_number': pitch_num,
        'at_bat_id': int(np.random.randint(1000000, 9999999))
    }
```

### Injury Scenario Generation

```python
def _generate_injury_scenario(self, pitchers, appearances):
    """Generate realistic injury scenario for demonstration."""
    
    # Select one pitcher for injury scenario
    target_pitcher = pitchers[1]  # Carlos Rodón (second pitcher)
    
    # Find the pitcher's appearances
    pitcher_appearances = appearances[
        appearances['pitcher_id'] == target_pitcher['pitcher_id']
    ].sort_values('game_date')
    
    # Create injury scenario: velocity decline followed by injury
    injury_date = None
    velocity_decline_start = None
    
    # Find a good point for injury scenario (around 2/3 through the season)
    mid_point = len(pitcher_appearances) // 3 * 2
    if mid_point < len(pitcher_appearances):
        # Start velocity decline 3-4 appearances before injury
        decline_start_idx = max(0, mid_point - 4)
        velocity_decline_start = pitcher_appearances.iloc[decline_start_idx]['game_date']
        
        # Injury occurs 10-15 days after velocity decline starts
        injury_date = velocity_decline_start + pd.Timedelta(days=np.random.randint(10, 16))
        
        # Ensure injury date is within our date range
        if injury_date > pitcher_appearances.iloc[-1]['game_date']:
            injury_date = pitcher_appearances.iloc[-1]['game_date'] - pd.Timedelta(days=5)
    
    if injury_date:
        injury = {
            'pitcher_id': target_pitcher['pitcher_id'],
            'il_start': injury_date.strftime('%Y-%m-%d'),
            'il_end': (injury_date + pd.Timedelta(days=45)).strftime('%Y-%m-%d'),  # 45-day IL stint
            'stint_type': 'Elbow inflammation'
        }
        
        return [injury]
    
    return []

def _apply_injury_pattern(self, pitcher_appearances, decline_start_date, injury_date):
    """Apply velocity decline pattern leading to injury."""
    
    # Modify appearances to show velocity decline
    modified_appearances = pitcher_appearances.copy()
    
    for idx, row in modified_appearances.iterrows():
        appearance_date = pd.to_datetime(row['game_date'])
        
        if appearance_date >= decline_start_date:
            # Calculate days since decline started
            days_since_decline = (appearance_date - decline_start_date).days
            
            # Apply progressive velocity decline
            if days_since_decline <= 7:
                # Initial decline: -0.5 to -1.0 MPH
                decline_factor = np.random.uniform(-1.0, -0.5)
            elif days_since_decline <= 14:
                # Accelerated decline: -1.0 to -2.0 MPH
                decline_factor = np.random.uniform(-2.0, -1.0)
            else:
                # Severe decline: -2.0 to -3.0 MPH
                decline_factor = np.random.uniform(-3.0, -2.0)
            
            # Apply decline to velocity
            original_vel = row['avg_vel']
            new_vel = original_vel + decline_factor
            modified_appearances.at[idx, 'avg_vel'] = max(88, new_vel)  # Don't go below 88 MPH
    
    return modified_appearances
```

## Test Data Specifications

### Unit Test Fixtures

```python
def create_unit_test_fixtures():
    """Create minimal fixtures for unit tests."""
    
    # Minimal pitch data
    test_pitches = pd.DataFrame({
        'game_date': ['2024-04-01', '2024-04-01', '2024-04-01'],
        'pitcher_id': [12345, 12345, 12345],
        'pitch_type': ['FF', 'SL', 'CH'],
        'release_speed': [94.5, 88.2, 85.1],
        'release_spin_rate': [2450, 2600, 1800],
        'pitch_number': [1, 2, 3],
        'at_bat_id': [123456789, 123456789, 123456790]
    })
    
    # Minimal appearance data
    test_appearances = pd.DataFrame({
        'game_date': ['2024-04-01', '2024-04-05', '2024-04-10'],
        'pitcher_id': [12345, 12345, 12345],
        'pitches_thrown': [85, 95, 90],
        'avg_vel': [93.2, 94.1, 92.8],
        'avg_spin': [2350, 2400, 2300],
        'outs_recorded': [15, 18, 16],
        'innings_pitched': [5.0, 6.0, 5.5]
    })
    
    # Minimal injury data
    test_injuries = pd.DataFrame({
        'pitcher_id': [12345],
        'il_start': ['2024-04-15'],
        'il_end': ['2024-05-30'],
        'stint_type': ['Elbow inflammation']
    })
    
    return {
        'pitches': test_pitches,
        'appearances': test_appearances,
        'injuries': test_injuries
    }
```

### Integration Test Fixtures

```python
def create_integration_test_fixtures():
    """Create comprehensive fixtures for integration tests."""
    
    # Generate 2 pitchers with 30 days of data
    generator = MockDataGenerator({'random_seed': 123})
    
    test_data = generator.generate_complete_dataset(
        num_pitchers=2,
        start_date='2024-04-01',
        end_date='2024-04-30',
        injury_scenario=False  # No injury for integration tests
    )
    
    return test_data

def create_edge_case_fixtures():
    """Create fixtures for edge case testing."""
    
    # Pitcher with very high workload
    high_workload_pitcher = {
        'pitcher_id': 99999,
        'name': 'High Workload Pitcher',
        'role': 'starter',
        'baseline_velocity': 95.0,
        'baseline_spin_rate': 2400
    }
    
    # Generate high workload pattern
    high_workload_pitches = []
    for day in range(1, 8):  # 7 consecutive days
        game_date = f'2024-04-{day:02d}'
        for pitch_num in range(1, 121):  # 120 pitches per game
            pitch = {
                'game_date': game_date,
                'pitcher_id': 99999,
                'pitch_type': 'FF',
                'release_speed': 95.0,
                'release_spin_rate': 2400,
                'pitch_number': pitch_num,
                'at_bat_id': 999999999
            }
            high_workload_pitches.append(pitch)
    
    # Pitcher with velocity decline
    declining_pitcher = {
        'pitcher_id': 88888,
        'name': 'Declining Velocity Pitcher',
        'role': 'starter',
        'baseline_velocity': 95.0,
        'baseline_spin_rate': 2400
    }
    
    # Generate velocity decline pattern
    declining_pitches = []
    for day in range(1, 16):  # 15 days of decline
        game_date = f'2024-04-{day:02d}'
        # Progressive velocity decline
        velocity = 95.0 - (day * 0.2)  # 0.2 MPH decline per day
        
        for pitch_num in range(1, 96):  # 95 pitches per game
            pitch = {
                'game_date': game_date,
                'pitcher_id': 88888,
                'pitch_type': 'FF',
                'release_speed': max(88, velocity + np.random.normal(0, 1)),
                'release_spin_rate': 2400,
                'pitch_number': pitch_num,
                'at_bat_id': 888888888
            }
            declining_pitches.append(pitch)
    
    return {
        'high_workload_pitcher': high_workload_pitcher,
        'high_workload_pitches': pd.DataFrame(high_workload_pitches),
        'declining_pitcher': declining_pitcher,
        'declining_pitches': pd.DataFrame(declining_pitches)
    }
```

### Performance Test Fixtures

```python
def create_performance_test_fixtures():
    """Create large datasets for performance testing."""
    
    # Generate 100 pitchers with 90 days of data
    generator = MockDataGenerator({'random_seed': 456})
    
    performance_data = generator.generate_complete_dataset(
        num_pitchers=100,
        start_date='2024-04-01',
        end_date='2024-06-30',
        injury_scenario=False
    )
    
    return performance_data

def create_stress_test_fixtures():
    """Create extreme datasets for stress testing."""
    
    # Generate 1000 pitchers with 365 days of data
    generator = MockDataGenerator({'random_seed': 789})
    
    stress_data = generator.generate_complete_dataset(
        num_pitchers=1000,
        start_date='2023-01-01',
        end_date='2023-12-31',
        injury_scenario=False
    )
    
    return stress_data
```

## Demo Data Specifications

### Demo Scenario 1: Normal Operations

```python
def create_normal_demo_data():
    """Create demo data showing normal operations."""
    
    generator = MockDataGenerator({'random_seed': 42})
    
    demo_data = generator.generate_complete_dataset(
        num_pitchers=5,
        start_date='2024-04-01',
        end_date='2024-06-30',
        injury_scenario=False
    )
    
    # Add some normal risk variations
    demo_data = _add_normal_risk_variations(demo_data)
    
    return demo_data

def _add_normal_risk_variations(data):
    """Add normal risk variations to demo data."""
    
    appearances = data['appearances'].copy()
    
    # Add some pitchers with elevated workload
    high_workload_pitchers = [10000, 10002]  # First and third pitchers
    
    for pitcher_id in high_workload_pitchers:
        pitcher_appearances = appearances[appearances['pitcher_id'] == pitcher_id]
        
        # Increase pitch counts for some appearances
        for idx in pitcher_appearances.index:
            if np.random.random() < 0.3:  # 30% chance
                appearances.at[idx, 'pitches_thrown'] += np.random.randint(10, 25)
    
    data['appearances'] = appearances
    return data
```

### Demo Scenario 2: Injury Risk Detection

```python
def create_injury_demo_data():
    """Create demo data with clear injury risk scenario."""
    
    generator = MockDataGenerator({'random_seed': 123})
    
    demo_data = generator.generate_complete_dataset(
        num_pitchers=3,
        start_date='2024-04-01',
        end_date='2024-06-30',
        injury_scenario=True
    )
    
    # Enhance the injury scenario
    demo_data = _enhance_injury_scenario(demo_data)
    
    return demo_data

def _enhance_injury_scenario(data):
    """Enhance the injury scenario for better demo."""
    
    # Find the pitcher with injury
    injury_pitcher_id = data['injuries'][0]['pitcher_id']
    
    # Modify appearances to show clear risk patterns
    appearances = data['appearances'].copy()
    pitcher_appearances = appearances[appearances['pitcher_id'] == injury_pitcher_id]
    
    # Add high workload pattern before injury
    injury_date = pd.to_datetime(data['injuries'][0]['il_start'])
    
    for idx in pitcher_appearances.index:
        appearance_date = pd.to_datetime(pitcher_appearances.at[idx, 'game_date'])
        days_before_injury = (injury_date - appearance_date).days
        
        if 0 <= days_before_injury <= 21:  # 21 days before injury
            # Increase pitch count
            appearances.at[idx, 'pitches_thrown'] += np.random.randint(15, 30)
            
            # Decrease velocity
            appearances.at[idx, 'avg_vel'] -= np.random.uniform(1.0, 2.5)
    
    data['appearances'] = appearances
    return data
```

### Demo Scenario 3: Multiple Risk Patterns

```python
def create_multiple_risk_demo_data():
    """Create demo data with multiple risk patterns."""
    
    generator = MockDataGenerator({'random_seed': 456})
    
    demo_data = generator.generate_complete_dataset(
        num_pitchers=8,
        start_date='2024-04-01',
        end_date='2024-06-30',
        injury_scenario=False
    )
    
    # Add different risk patterns
    demo_data = _add_multiple_risk_patterns(demo_data)
    
    return demo_data

def _add_multiple_risk_patterns(data):
    """Add multiple different risk patterns."""
    
    appearances = data['appearances'].copy()
    
    # Pattern 1: High workload pitcher
    high_workload_pitcher = 10000
    pitcher_appearances = appearances[appearances['pitcher_id'] == high_workload_pitcher]
    for idx in pitcher_appearances.index:
        appearances.at[idx, 'pitches_thrown'] += np.random.randint(20, 35)
    
    # Pattern 2: Velocity decline pitcher
    declining_pitcher = 10001
    pitcher_appearances = appearances[appearances['pitcher_id'] == declining_pitcher]
    for idx in pitcher_appearances.index:
        appearances.at[idx, 'avg_vel'] -= np.random.uniform(0.5, 1.5)
    
    # Pattern 3: Short rest pitcher
    short_rest_pitcher = 10002
    # This will be handled by the appearance schedule generation
    
    # Pattern 4: Spin rate decline pitcher
    spin_decline_pitcher = 10003
    pitcher_appearances = appearances[appearances['pitcher_id'] == spin_decline_pitcher]
    for idx in pitcher_appearances.index:
        appearances.at[idx, 'avg_spin'] -= np.random.randint(50, 150)
    
    data['appearances'] = appearances
    return data
```

## Data Validation

### Mock Data Quality Checks

```python
def validate_mock_data(data):
    """Validate generated mock data quality."""
    
    validation_results = {
        'pitches': validate_pitch_data(data['pitches']),
        'appearances': validate_appearance_data(data['appearances']),
        'injuries': validate_injury_data(data['injuries'])
    }
    
    return validation_results

def validate_pitch_data(pitches_df):
    """Validate pitch data quality."""
    
    issues = []
    
    # Check velocity range
    invalid_velocities = pitches_df[
        (pitches_df['release_speed'] < 85) | (pitches_df['release_speed'] > 105)
    ]
    if len(invalid_velocities) > 0:
        issues.append(f"Found {len(invalid_velocities)} pitches with invalid velocities")
    
    # Check spin rate range
    invalid_spin_rates = pitches_df[
        (pitches_df['release_spin_rate'] < 1800) | (pitches_df['release_spin_rate'] > 3200)
    ]
    if len(invalid_spin_rates) > 0:
        issues.append(f"Found {len(invalid_spin_rates)} pitches with invalid spin rates")
    
    # Check for missing values
    missing_values = pitches_df.isnull().sum()
    if missing_values.sum() > 0:
        issues.append(f"Found missing values: {missing_values.to_dict()}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'total_pitches': len(pitches_df),
        'unique_pitchers': pitches_df['pitcher_id'].nunique()
    }

def validate_appearance_data(appearances_df):
    """Validate appearance data quality."""
    
    issues = []
    
    # Check pitch count ranges
    invalid_pitch_counts = appearances_df[
        (appearances_df['pitches_thrown'] < 1) | (appearances_df['pitches_thrown'] > 150)
    ]
    if len(invalid_pitch_counts) > 0:
        issues.append(f"Found {len(invalid_pitch_counts)} appearances with invalid pitch counts")
    
    # Check velocity ranges
    invalid_velocities = appearances_df[
        (appearances_df['avg_vel'] < 85) | (appearances_df['avg_vel'] > 105)
    ]
    if len(invalid_velocities) > 0:
        issues.append(f"Found {len(invalid_velocities)} appearances with invalid velocities")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'total_appearances': len(appearances_df),
        'unique_pitchers': appearances_df['pitcher_id'].nunique()
    }
```

## Data Export and Persistence

### Export Formats

```python
def export_mock_data(data, output_dir, format='csv'):
    """Export mock data to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if format == 'csv':
        # Export to CSV files
        data['pitches'].to_csv(f'{output_dir}/pitches.csv', index=False)
        data['appearances'].to_csv(f'{output_dir}/appearances.csv', index=False)
        data['injuries'].to_csv(f'{output_dir}/injuries.csv', index=False)
        
        # Create metadata file
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'random_seed': 42,
            'num_pitchers': data['pitches']['pitcher_id'].nunique(),
            'date_range': {
                'start': data['pitches']['game_date'].min(),
                'end': data['pitches']['game_date'].max()
            },
            'total_pitches': len(data['pitches']),
            'total_appearances': len(data['appearances']),
            'total_injuries': len(data['injuries'])
        }
        
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    elif format == 'parquet':
        # Export to Parquet files
        data['pitches'].to_parquet(f'{output_dir}/pitches.parquet', index=False)
        data['appearances'].to_parquet(f'{output_dir}/appearances.parquet', index=False)
        data['injuries'].to_parquet(f'{output_dir}/injuries.parquet', index=False)

def load_mock_data(input_dir, format='csv'):
    """Load mock data from files."""
    
    if format == 'csv':
        data = {
            'pitches': pd.read_csv(f'{input_dir}/pitches.csv'),
            'appearances': pd.read_csv(f'{input_dir}/appearances.csv'),
            'injuries': pd.read_csv(f'{input_dir}/injuries.csv')
        }
    elif format == 'parquet':
        data = {
            'pitches': pd.read_parquet(f'{input_dir}/pitches.parquet'),
            'appearances': pd.read_parquet(f'{input_dir}/appearances.parquet'),
            'injuries': pd.read_parquet(f'{input_dir}/injuries.parquet')
        }
    
    return data
```

## Usage Examples

### Command Line Interface

```python
def main():
    """Command line interface for mock data generation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate mock data for PitchGuard')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--num-pitchers', type=int, default=3, help='Number of pitchers')
    parser.add_argument('--start-date', default='2024-04-01', help='Start date')
    parser.add_argument('--end-date', default='2024-06-30', help='End date')
    parser.add_argument('--injury-scenario', action='store_true', help='Include injury scenario')
    parser.add_argument('--format', choices=['csv', 'parquet'], default='csv', help='Output format')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Generate data
    generator = MockDataGenerator({'random_seed': args.random_seed})
    data = generator.generate_complete_dataset(
        num_pitchers=args.num_pitchers,
        start_date=args.start_date,
        end_date=args.end_date,
        injury_scenario=args.injury_scenario
    )
    
    # Export data
    export_mock_data(data, args.output_dir, args.format)
    
    # Validate data
    validation_results = validate_mock_data(data)
    
    print(f"Generated mock data:")
    print(f"- {validation_results['pitches']['total_pitches']} pitches")
    print(f"- {validation_results['appearances']['total_appearances']} appearances")
    print(f"- {len(data['injuries'])} injuries")
    print(f"- {validation_results['pitches']['unique_pitchers']} pitchers")
    
    if all(result['valid'] for result in validation_results.values()):
        print("✓ All data validation checks passed")
    else:
        print("✗ Some validation issues found:")
        for data_type, result in validation_results.items():
            if not result['valid']:
                print(f"  {data_type}: {result['issues']}")

if __name__ == '__main__':
    main()
```

### Python API Usage

```python
# Example usage in tests
def test_feature_engineering():
    """Test feature engineering with mock data."""
    
    # Generate test data
    generator = MockDataGenerator({'random_seed': 42})
    test_data = generator.generate_complete_dataset(
        num_pitchers=2,
        start_date='2024-04-01',
        end_date='2024-04-30',
        injury_scenario=False
    )
    
    # Test feature engineering
    features = compute_features(test_data['appearances'])
    
    # Assert expected results
    assert len(features) > 0
    assert 'roll7d_pitch_count' in features.columns
    assert features['roll7d_pitch_count'].min() >= 0

# Example usage in demo
def run_demo():
    """Run demo with mock data."""
    
    # Generate demo data with injury scenario
    generator = MockDataGenerator({'random_seed': 123})
    demo_data = generator.generate_complete_dataset(
        num_pitchers=3,
        start_date='2024-04-01',
        end_date='2024-06-30',
        injury_scenario=True
    )
    
    # Load data into system
    load_data_into_system(demo_data)
    
    # Run demo scenarios
    run_demo_scenarios()
```
