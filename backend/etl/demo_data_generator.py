#!/usr/bin/env python3
"""
Demo Data Generator for PitchGuard

Creates different scenarios for demonstration purposes:
1. Normal operations (no injuries)
2. Injury risk detection (clear injury scenario)
3. Multiple risk patterns (various risk factors)
"""

import os
import sys
from datetime import datetime
from mock_data_generator import MockDataGenerator, export_mock_data


def create_normal_demo_data():
    """Create demo data showing normal operations."""
    print("Creating normal demo data...")
    
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


def create_injury_demo_data():
    """Create demo data with clear injury risk scenario."""
    print("Creating injury demo data...")
    
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


def create_multiple_risk_demo_data():
    """Create demo data with multiple risk patterns."""
    print("Creating multiple risk patterns demo data...")
    
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


def _add_normal_risk_variations(data):
    """Add normal risk variations to demo data."""
    
    appearances = data['appearances'].copy()
    
    # Add some pitchers with elevated workload
    high_workload_pitchers = [10000, 10002]  # First and third pitchers
    
    for pitcher_id in high_workload_pitchers:
        pitcher_appearances = appearances[appearances['pitcher_id'] == pitcher_id]
        
        # Increase pitch counts for some appearances
        for idx in pitcher_appearances.index:
            if appearances.at[idx, 'pitches_thrown'] < 100:  # Only increase if not already high
                appearances.at[idx, 'pitches_thrown'] += 15
    
    data['appearances'] = appearances
    return data


def _enhance_injury_scenario(data):
    """Enhance the injury scenario for better demo."""
    
    if not data['injuries']:
        return data
    
    # Find the pitcher with injury
    injury_pitcher_id = data['injuries'][0]['pitcher_id']
    
    # Modify appearances to show clear risk patterns
    appearances = data['appearances'].copy()
    pitcher_appearances = appearances[appearances['pitcher_id'] == injury_pitcher_id]
    
    # Add high workload pattern before injury
    injury_date = datetime.strptime(data['injuries'][0]['il_start'], '%Y-%m-%d')
    
    for idx in pitcher_appearances.index:
        appearance_date = datetime.strptime(pitcher_appearances.at[idx, 'game_date'], '%Y-%m-%d')
        days_before_injury = (injury_date - appearance_date).days
        
        if 0 <= days_before_injury <= 21:  # 21 days before injury
            # Increase pitch count
            appearances.at[idx, 'pitches_thrown'] += 20
            
            # Decrease velocity slightly
            appearances.at[idx, 'avg_vel'] -= 1.5
    
    data['appearances'] = appearances
    return data


def _add_multiple_risk_patterns(data):
    """Add multiple different risk patterns."""
    
    appearances = data['appearances'].copy()
    
    # Pattern 1: High workload pitcher (first pitcher)
    high_workload_pitcher = 10000
    pitcher_appearances = appearances[appearances['pitcher_id'] == high_workload_pitcher]
    for idx in pitcher_appearances.index:
        appearances.at[idx, 'pitches_thrown'] += 25
    
    # Pattern 2: Velocity decline pitcher (second pitcher)
    declining_pitcher = 10001
    pitcher_appearances = appearances[appearances['pitcher_id'] == declining_pitcher]
    for idx in pitcher_appearances.index:
        appearances.at[idx, 'avg_vel'] -= 2.0
    
    # Pattern 3: Spin rate decline pitcher (third pitcher)
    spin_decline_pitcher = 10002
    pitcher_appearances = appearances[appearances['pitcher_id'] == spin_decline_pitcher]
    for idx in pitcher_appearances.index:
        appearances.at[idx, 'avg_spin'] -= 200
    
    # Pattern 4: Inconsistent performance pitcher (fourth pitcher)
    inconsistent_pitcher = 10003
    pitcher_appearances = appearances[appearances['pitcher_id'] == inconsistent_pitcher]
    for idx in pitcher_appearances.index:
        # Add high variance to velocity
        appearances.at[idx, 'vel_std'] += 1.0
    
    data['appearances'] = appearances
    return data


def main():
    """Generate all demo scenarios."""
    
    # Create output directory
    output_base = "../data/demo"
    os.makedirs(output_base, exist_ok=True)
    
    # Generate normal demo data
    print("=" * 50)
    print("Generating Normal Demo Data")
    print("=" * 50)
    normal_data = create_normal_demo_data()
    export_mock_data(normal_data, f"{output_base}/normal", format='csv')
    print(f"✓ Normal demo data exported to {output_base}/normal/")
    
    # Generate injury demo data
    print("\n" + "=" * 50)
    print("Generating Injury Demo Data")
    print("=" * 50)
    injury_data = create_injury_demo_data()
    export_mock_data(injury_data, f"{output_base}/injury", format='csv')
    print(f"✓ Injury demo data exported to {output_base}/injury/")
    
    # Generate multiple risk patterns demo data
    print("\n" + "=" * 50)
    print("Generating Multiple Risk Patterns Demo Data")
    print("=" * 50)
    risk_data = create_multiple_risk_demo_data()
    export_mock_data(risk_data, f"{output_base}/multiple_risks", format='csv')
    print(f"✓ Multiple risk patterns demo data exported to {output_base}/multiple_risks/")
    
    print("\n" + "=" * 50)
    print("🎉 All demo scenarios generated successfully!")
    print("=" * 50)
    print(f"Demo data available in: {output_base}/")
    print("- normal/: Normal operations scenario")
    print("- injury/: Injury risk detection scenario")
    print("- multiple_risks/: Multiple risk patterns scenario")


if __name__ == '__main__':
    main()
