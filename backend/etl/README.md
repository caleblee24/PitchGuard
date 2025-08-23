# Mock Data Generator

This directory contains tools for generating realistic mock data for PitchGuard development, testing, and demonstration.

## Files

- `mock_data_generator.py` - Main mock data generator
- `demo_data_generator.py` - Demo scenario generator
- `test_mock_data.py` - Test suite for the generator

## Quick Start

### Generate Basic Mock Data

```bash
# Generate 5 pitchers with injury scenario
python3 etl/mock_data_generator.py \
  --output-dir ../data/mock \
  --num-pitchers 5 \
  --start-date 2024-04-01 \
  --end-date 2024-06-30 \
  --injury-scenario
```

### Generate Demo Scenarios

```bash
# Generate all demo scenarios
python3 etl/demo_data_generator.py
```

This creates three scenarios in `../data/demo/`:
- `normal/` - Normal operations (no injuries)
- `injury/` - Clear injury risk scenario
- `multiple_risks/` - Various risk patterns

### Run Tests

```bash
# Run the test suite
python3 test_mock_data.py
```

## Generated Data

The mock data generator creates realistic MLB data with the following characteristics:

### Pitches Data (`pitches.csv`)
- **9,000+ pitches** per 5 pitchers over 3 months
- Realistic velocity ranges (85-105 MPH)
- Realistic spin rates (1800-3200 RPM)
- Multiple pitch types (FF, SL, CH, CT, CB, SI)
- Proper pitch sequencing and game dates

### Appearances Data (`appearances.csv`)
- **100+ appearances** aggregated from pitch data
- Pitch counts, average velocity/spin rates
- Velocity and spin rate standard deviations
- Simulated outs and innings pitched

### Injuries Data (`injuries.csv`)
- **1 injury scenario** with realistic patterns
- 45-day IL stint with "Elbow inflammation"
- Velocity decline pattern leading to injury

### Metadata (`metadata.json`)
- Generation timestamp and random seed
- Data summary statistics
- Date ranges and counts

## Data Quality

The generator includes comprehensive validation:

- ✅ Velocity ranges (85-105 MPH)
- ✅ Spin rate ranges (1800-3200 RPM)
- ✅ Pitch count ranges (1-150 per appearance)
- ✅ Realistic pitcher roles (70% starters, 30% relievers)
- ✅ Proper date sequencing
- ✅ No missing values

## Realistic Patterns

### Pitcher Profiles
- **Starters**: 92-97 MPH baseline, 95±15 pitches per game
- **Relievers**: 93-98 MPH baseline, 25±10 pitches per game
- **Injury Risk**: 60% low, 30% medium, 10% high

### Injury Scenarios
- Velocity decline over 7-14 days before injury
- Progressive decline pattern (initial → accelerated → severe)
- Realistic 45-day IL stint duration

### Workload Patterns
- Starters: Every 4-5 days with variation
- Relievers: Every 1-3 days with irregular intervals
- Realistic pitch count variations

## Command Line Options

### mock_data_generator.py

```
--output-dir DIR        Output directory (required)
--num-pitchers N        Number of pitchers (default: 3)
--start-date DATE       Start date YYYY-MM-DD (default: 2024-04-01)
--end-date DATE         End date YYYY-MM-DD (default: 2024-06-30)
--injury-scenario       Include injury scenario
--format FORMAT         Output format: csv or parquet (default: csv)
--random-seed SEED      Random seed for reproducibility (default: 42)
```

## Use Cases

### Development
```bash
# Quick development data
python3 etl/mock_data_generator.py --output-dir ./dev_data --num-pitchers 2
```

### Testing
```bash
# Test data with known patterns
python3 etl/mock_data_generator.py --output-dir ./test_data --random-seed 123
```

### Demo
```bash
# Generate all demo scenarios
python3 etl/demo_data_generator.py
```

### Production Simulation
```bash
# Larger dataset for performance testing
python3 etl/mock_data_generator.py \
  --output-dir ./prod_sim \
  --num-pitchers 50 \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --injury-scenario
```

## Data Schema

### Pitches Table
| Field | Type | Description |
|-------|------|-------------|
| game_date | DATE | Game date |
| pitcher_id | INT | Unique pitcher identifier |
| pitch_type | STR | Pitch type (FF, SL, CH, etc.) |
| release_speed | FLOAT | Pitch velocity in MPH |
| release_spin_rate | INT | Spin rate in RPM |
| pitch_number | INT | Pitch number in appearance |
| at_bat_id | INT | Unique at-bat identifier |

### Appearances Table
| Field | Type | Description |
|-------|------|-------------|
| game_date | DATE | Game date |
| pitcher_id | INT | Unique pitcher identifier |
| pitches_thrown | INT | Total pitches in appearance |
| avg_vel | FLOAT | Average velocity |
| avg_spin | INT | Average spin rate |
| vel_std | FLOAT | Velocity standard deviation |
| spin_std | INT | Spin rate standard deviation |
| outs_recorded | INT | Outs recorded |
| innings_pitched | FLOAT | Innings pitched |

### Injuries Table
| Field | Type | Description |
|-------|------|-------------|
| pitcher_id | INT | Unique pitcher identifier |
| il_start | DATE | IL start date |
| il_end | DATE | IL end date |
| stint_type | STR | Injury description |

## Customization

The generator can be customized by modifying:

1. **Pitcher Profiles** - Baseline characteristics in `_load_pitcher_profiles()`
2. **Injury Patterns** - Risk patterns in `_load_injury_patterns()`
3. **Pitch Types** - Distribution in `_generate_single_pitch()`
4. **Workload Patterns** - Scheduling in `_generate_appearance_schedule()`

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure output directory is writable
2. **Memory Issues**: Reduce `--num-pitchers` for large date ranges
3. **No Injury Generated**: Increase date range or number of pitchers

### Validation

The generator includes built-in validation. Check the output for:
- ✓ All validation checks passed
- Realistic data ranges
- Proper file generation

If validation fails, check the error messages for specific issues.
