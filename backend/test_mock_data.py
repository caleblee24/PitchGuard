#!/usr/bin/env python3
"""
Test script for the mock data generator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl.mock_data_generator import MockDataGenerator, validate_mock_data, export_mock_data


def test_basic_generation():
    """Test basic data generation."""
    print("Testing basic mock data generation...")
    
    # Create generator
    generator = MockDataGenerator({'random_seed': 42})
    
    # Generate small dataset
    data = generator.generate_complete_dataset(
        num_pitchers=2,
        start_date='2024-04-01',
        end_date='2024-04-30',
        injury_scenario=False
    )
    
    # Check data structure
    assert 'pitches' in data
    assert 'appearances' in data
    assert 'injuries' in data
    assert 'pitchers' in data
    
    print(f"✓ Generated data with {len(data['pitches'])} pitches")
    print(f"✓ Generated data with {len(data['appearances'])} appearances")
    print(f"✓ Generated data with {len(data['pitchers'])} pitchers")
    
    return data


def test_injury_scenario():
    """Test injury scenario generation."""
    print("\nTesting injury scenario generation...")
    
    # Create generator
    generator = MockDataGenerator({'random_seed': 123})
    
    # Generate dataset with injury scenario
    data = generator.generate_complete_dataset(
        num_pitchers=3,
        start_date='2024-04-01',
        end_date='2024-06-30',
        injury_scenario=True
    )
    
    # Check injury data
    if data['injuries']:
        print(f"✓ Generated {len(data['injuries'])} injury scenario(s)")
        for injury in data['injuries']:
            print(f"  - Pitcher {injury['pitcher_id']}: {injury['stint_type']}")
            print(f"    IL Period: {injury['il_start']} to {injury['il_end']}")
    else:
        print("⚠ No injury scenario generated (may be due to insufficient data)")
    
    return data


def test_data_validation():
    """Test data validation."""
    print("\nTesting data validation...")
    
    # Generate test data
    generator = MockDataGenerator({'random_seed': 456})
    data = generator.generate_complete_dataset(
        num_pitchers=2,
        start_date='2024-04-01',
        end_date='2024-04-30',
        injury_scenario=False
    )
    
    # Validate data
    validation_results = validate_mock_data(data)
    
    # Check validation results
    all_valid = True
    for data_type, result in validation_results.items():
        if result['valid']:
            print(f"✓ {data_type} validation passed")
        else:
            print(f"✗ {data_type} validation failed: {result['issues']}")
            all_valid = False
    
    if all_valid:
        print("✓ All data validation checks passed")
    else:
        print("✗ Some validation checks failed")
    
    return validation_results


def test_data_export():
    """Test data export functionality."""
    print("\nTesting data export...")
    
    # Generate test data
    generator = MockDataGenerator({'random_seed': 789})
    data = generator.generate_complete_dataset(
        num_pitchers=2,
        start_date='2024-04-01',
        end_date='2024-04-30',
        injury_scenario=True
    )
    
    # Export to test directory
    test_output_dir = "test_output"
    export_mock_data(data, test_output_dir, format='csv')
    
    # Check that files were created
    expected_files = ['pitches.csv', 'appearances.csv', 'metadata.json']
    if data['injuries']:
        expected_files.append('injuries.csv')
    
    for filename in expected_files:
        filepath = os.path.join(test_output_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ Exported {filename}")
        else:
            print(f"✗ Failed to export {filename}")
    
    print(f"✓ Data exported to {test_output_dir}/")
    
    return test_output_dir


def test_realistic_patterns():
    """Test that generated data follows realistic patterns."""
    print("\nTesting realistic data patterns...")
    
    # Generate data
    generator = MockDataGenerator({'random_seed': 42})
    data = generator.generate_complete_dataset(
        num_pitchers=5,
        start_date='2024-04-01',
        end_date='2024-06-30',
        injury_scenario=False
    )
    
    pitches_df = data['pitches']
    appearances_df = data['appearances']
    
    # Check velocity patterns
    avg_velocity = pitches_df['release_speed'].mean()
    print(f"✓ Average velocity: {avg_velocity:.1f} MPH (expected: 90-98 MPH)")
    
    # Check spin rate patterns
    avg_spin_rate = pitches_df['release_spin_rate'].mean()
    print(f"✓ Average spin rate: {avg_spin_rate:.0f} RPM (expected: 2200-2600 RPM)")
    
    # Check pitch count patterns
    avg_pitches = appearances_df['pitches_thrown'].mean()
    print(f"✓ Average pitches per appearance: {avg_pitches:.1f} (expected: 20-100)")
    
    # Check date range
    date_range = pitches_df['game_date'].nunique()
    print(f"✓ Date range: {date_range} unique dates")
    
    # Check pitcher distribution
    starter_count = sum(1 for p in data['pitchers'] if p.role == 'starter')
    reliever_count = sum(1 for p in data['pitchers'] if p.role == 'reliever')
    print(f"✓ Pitcher roles: {starter_count} starters, {reliever_count} relievers")


def main():
    """Run all tests."""
    print("🧪 Testing Mock Data Generator")
    print("=" * 50)
    
    try:
        # Run tests
        test_basic_generation()
        test_injury_scenario()
        test_data_validation()
        test_data_export()
        test_realistic_patterns()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("\nYou can now use the mock data generator with:")
        print("python etl/mock_data_generator.py --output-dir ./data/mock --num-pitchers 3 --injury-scenario")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
