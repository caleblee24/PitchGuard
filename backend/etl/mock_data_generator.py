"""
Mock Data Generator for PitchGuard

Generates realistic MLB pitch data, appearances, and injury scenarios for development,
testing, and demonstration purposes.
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class PitcherProfile:
    """Pitcher profile with baseline characteristics."""
    pitcher_id: int
    name: str
    team: str
    role: str  # 'starter' or 'reliever'
    baseline_velocity: float
    baseline_spin_rate: int
    injury_risk_profile: str  # 'low', 'medium', 'high'


class MockDataGenerator:
    """Generate realistic mock data for PitchGuard."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.random_seed = config.get('random_seed', 42)
        self.pitcher_profiles = self._load_pitcher_profiles()
        self.injury_patterns = self._load_injury_patterns()
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
    
    def _load_pitcher_profiles(self) -> Dict:
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
    
    def _load_injury_patterns(self) -> Dict:
        """Load injury pattern configurations."""
        return {
            'velocity_decline': {
                'initial_days': 7,
                'initial_decline': (-1.0, -0.5),
                'accelerated_days': 14,
                'accelerated_decline': (-2.0, -1.0),
                'severe_decline': (-3.0, -2.0)
            },
            'workload_increase': {
                'pitch_count_increase': (15, 30),
                'frequency_increase': 0.3
            }
        }
    
    def generate_complete_dataset(
        self, 
        num_pitchers: int = 3, 
        start_date: str = '2024-04-01',
        end_date: str = '2024-06-30',
        injury_scenario: bool = True
    ) -> Dict:
        """Generate complete mock dataset."""
        
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
    
    def generate_realistic_injury_scenarios(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
        """Generate data with more realistic injury patterns for comprehensive testing."""
        
        # Use configuration from self.config
        num_pitchers = self.config.get('num_pitchers', 20)
        start_date = self.config.get('start_date', datetime.now() - timedelta(days=90))
        end_date = self.config.get('end_date', datetime.now())
        
        # Convert datetime to string if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Generate complete dataset with higher injury probability
        data = self.generate_complete_dataset(
            num_pitchers=num_pitchers,
            start_date=start_date,
            end_date=end_date,
            injury_scenario=True
        )
        
        return data['pitches'], data['appearances'], data['injuries']
    
    def _generate_pitcher_profiles(self, num_pitchers: int) -> List[PitcherProfile]:
        """Generate realistic pitcher profiles."""
        
        pitcher_names = [
            "Gerrit Cole", "Carlos Rodón", "Nestor Cortes",
            "Luis Severino", "Domingo Germán", "Clarke Schmidt",
            "Marcus Stroman", "Jameson Taillon", "Jordan Montgomery",
            "Corey Kluber", "Rich Hill", "Chris Sale"
        ]
        
        teams = ["NYY", "LAD", "HOU", "ATL", "BOS", "CHC", "TOR", "SF", "MIL", "CLE", "MIN", "TB"]
        
        pitchers = []
        for i in range(num_pitchers):
            # Select role with realistic distribution
            role = np.random.choice(['starter', 'reliever'], p=[0.7, 0.3])
            
            # Generate baseline characteristics
            vel_profile = self.pitcher_profiles['velocity_baselines'][role]
            baseline_velocity = np.random.normal(vel_profile['mean'], vel_profile['std'])
            baseline_velocity = max(vel_profile['min'], min(vel_profile['max'], baseline_velocity))
            
            spin_profile = self.pitcher_profiles['spin_rate_baselines'][role]
            baseline_spin_rate = int(np.random.normal(spin_profile['mean'], spin_profile['std']))
            baseline_spin_rate = max(spin_profile['min'], min(spin_profile['max'], baseline_spin_rate))
            
            # Injury risk profile - more realistic distribution for better training
            # Higher proportion of medium/high risk pitchers for better injury scenarios
            injury_risk = np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2])
            
            pitcher = PitcherProfile(
                pitcher_id=10000 + i,
                name=pitcher_names[i % len(pitcher_names)],
                team=teams[i % len(teams)],
                role=role,
                baseline_velocity=round(baseline_velocity, 1),
                baseline_spin_rate=baseline_spin_rate,
                injury_risk_profile=injury_risk
            )
            pitchers.append(pitcher)
        
        return pitchers
    
    def _generate_pitch_data(
        self, 
        pitchers: List[PitcherProfile], 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
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
    
    def _generate_appearance_schedule(
        self, 
        pitcher: PitcherProfile, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[datetime]:
        """Generate realistic appearance schedule."""
        
        if pitcher.role == 'starter':
            # Starters: every 4-5 days
            interval_days = np.random.choice([4, 5], p=[0.6, 0.4])
        else:
            # Relievers: more frequent, irregular intervals
            interval_days = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        
        appearance_dates = []
        current_date = start_date
        
        while current_date <= end_date:
            # Add some randomness to interval
            actual_interval = int(interval_days + np.random.randint(-1, 2))
            current_date += timedelta(days=max(1, actual_interval))
            
            if current_date <= end_date:
                appearance_dates.append(current_date)
        
        return appearance_dates
    
    def _generate_game_pitches(self, pitcher: PitcherProfile, game_date: datetime) -> List[Dict]:
        """Generate pitches for a single game appearance."""
        
        # Determine number of pitches based on role and random variation
        pitch_profile = self.pitcher_profiles['pitch_count_patterns'][pitcher.role]
        pitch_count = int(np.random.normal(pitch_profile['mean'], pitch_profile['std']))
        pitch_count = max(pitch_profile['min'], min(pitch_profile['max'], pitch_count))
        
        # Generate individual pitches
        pitches = []
        for pitch_num in range(1, pitch_count + 1):
            pitch = self._generate_single_pitch(pitcher, game_date, pitch_num)
            pitches.append(pitch)
        
        return pitches
    
    def _generate_single_pitch(
        self, 
        pitcher: PitcherProfile, 
        game_date: datetime, 
        pitch_num: int
    ) -> Dict:
        """Generate a single pitch with realistic characteristics."""
        
        # Pitch types with realistic frequencies
        pitch_types = ['FF', 'SL', 'CH', 'CT', 'CB', 'SI']
        pitch_weights = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]  # Four-seam most common
        
        pitch_type = np.random.choice(pitch_types, p=pitch_weights)
        
        # Velocity with realistic variation
        velocity_variation = np.random.normal(0, 1.5)  # ±3 MPH typical variation
        release_speed = pitcher.baseline_velocity + velocity_variation
        release_speed = max(85, min(105, release_speed))  # Realistic range
        
        # Spin rate correlated with velocity and pitch type
        if pitch_type == 'FF':
            spin_multiplier = 1.0
        elif pitch_type == 'SL':
            spin_multiplier = 1.1
        elif pitch_type == 'CH':
            spin_multiplier = 0.9
        else:
            spin_multiplier = 1.05
        
        spin_variation = np.random.normal(0, 100)
        release_spin_rate = int(pitcher.baseline_spin_rate * spin_multiplier + spin_variation)
        release_spin_rate = max(1800, min(3200, release_spin_rate))
        
        return {
            'game_date': game_date.strftime('%Y-%m-%d'),
            'pitcher_id': pitcher.pitcher_id,
            'pitch_type': pitch_type,
            'release_speed': round(release_speed, 1),
            'release_spin_rate': release_spin_rate,
            'pitch_number': pitch_num,
            'at_bat_id': int(np.random.randint(1000000, 9999999))
        }
    
    def _generate_appearances(self, pitches_df: pd.DataFrame) -> pd.DataFrame:
        """Generate appearance-level data from pitch data."""
        
        appearances = []
        
        # Group by pitcher and game date
        for (pitcher_id, game_date), group in pitches_df.groupby(['pitcher_id', 'game_date']):
            # Calculate appearance statistics
            appearance = {
                'game_date': game_date,
                'pitcher_id': pitcher_id,
                'pitches_thrown': len(group),
                'avg_vel': round(group['release_speed'].mean(), 1),
                'avg_spin': int(group['release_spin_rate'].mean()),
                'vel_std': round(group['release_speed'].std(), 1),
                'spin_std': int(group['release_spin_rate'].std()),
                'outs_recorded': int(np.random.normal(15, 3)),  # Simulated outs
                'innings_pitched': round(len(group) / 15, 1)  # Rough estimate
            }
            
            # Ensure realistic ranges
            appearance['outs_recorded'] = max(3, min(27, appearance['outs_recorded']))
            appearance['innings_pitched'] = max(0.1, min(9.0, appearance['innings_pitched']))
            
            appearances.append(appearance)
        
        return pd.DataFrame(appearances)
    
    def _generate_injury_scenario(
        self, 
        pitchers: List[PitcherProfile], 
        appearances: pd.DataFrame
    ) -> List[Dict]:
        """Generate realistic injury scenarios for demonstration."""
        
        injuries = []
        target_injury_rate = 0.08  # 8% injury rate (higher than current 0.17%)
        
        # Calculate how many injuries we want
        total_pitchers = len(pitchers)
        target_injuries = max(1, int(total_pitchers * target_injury_rate))
        
        # Prioritize pitchers with higher injury risk profiles
        high_risk_pitchers = [p for p in pitchers if p.injury_risk_profile == 'high']
        medium_risk_pitchers = [p for p in pitchers if p.injury_risk_profile == 'medium']
        low_risk_pitchers = [p for p in pitchers if p.injury_risk_profile == 'low']
        
        # Select pitchers for injuries (prioritize high risk)
        pitchers_to_injure = []
        pitchers_to_injure.extend(high_risk_pitchers[:min(len(high_risk_pitchers), target_injuries)])
        
        remaining_slots = target_injuries - len(pitchers_to_injure)
        if remaining_slots > 0:
            pitchers_to_injure.extend(medium_risk_pitchers[:min(len(medium_risk_pitchers), remaining_slots)])
        
        remaining_slots = target_injuries - len(pitchers_to_injure)
        if remaining_slots > 0:
            pitchers_to_injure.extend(low_risk_pitchers[:min(len(low_risk_pitchers), remaining_slots)])
        
        # Generate injury scenarios for selected pitchers
        for pitcher in pitchers_to_injure:
            injury = self._generate_single_injury_scenario(pitcher, appearances)
            if injury:
                injuries.append(injury)
        
        return injuries
    
    def _generate_single_injury_scenario(
        self, 
        pitcher: PitcherProfile, 
        appearances: pd.DataFrame
    ) -> Optional[Dict]:
        """Generate a single realistic injury scenario for a pitcher."""
        
        # Find the pitcher's appearances
        pitcher_appearances = appearances[
            appearances['pitcher_id'] == pitcher.pitcher_id
        ].sort_values('game_date')
        
        if len(pitcher_appearances) < 8:
            return None  # Not enough data for injury scenario
        
        # Create injury scenario: velocity decline followed by injury
        # Find a good point for injury scenario (around 2/3 through the season)
        mid_point = len(pitcher_appearances) // 3 * 2
        if mid_point < len(pitcher_appearances):
            # Start velocity decline 3-4 appearances before injury
            decline_start_idx = max(0, mid_point - 4)
            velocity_decline_start = pd.to_datetime(
                pitcher_appearances.iloc[decline_start_idx]['game_date']
            )
            
            # Injury occurs 10-15 days after velocity decline starts
            injury_date = velocity_decline_start + timedelta(
                days=int(np.random.randint(10, 16))
            )
            
            # Ensure injury date is within our date range
            last_appearance = pd.to_datetime(pitcher_appearances.iloc[-1]['game_date'])
            if injury_date > last_appearance:
                injury_date = last_appearance - timedelta(days=5)
            
            # Apply injury pattern to appearances
            self._apply_injury_pattern(pitcher_appearances, velocity_decline_start, injury_date)
            
            # Randomize injury type and duration
            injury_types = ['Elbow inflammation', 'Shoulder strain', 'Back tightness', 'Forearm soreness']
            injury_type = np.random.choice(injury_types)
            
            # Duration varies by injury type
            if 'elbow' in injury_type.lower():
                duration = np.random.randint(30, 60)  # 30-60 days
            elif 'shoulder' in injury_type.lower():
                duration = np.random.randint(45, 90)  # 45-90 days
            else:
                duration = np.random.randint(15, 45)  # 15-45 days
            
            injury = {
                'pitcher_id': pitcher.pitcher_id,
                'il_start': injury_date.strftime('%Y-%m-%d'),
                'il_end': (injury_date + timedelta(days=duration)).strftime('%Y-%m-%d'),
                'stint_type': injury_type
            }
            
            return injury
        
        return None
    
    def _apply_injury_pattern(
        self, 
        appearances: pd.DataFrame, 
        decline_start_date: datetime, 
        injury_date: datetime
    ) -> None:
        """Apply velocity decline pattern leading to injury."""
        
        for idx, row in appearances.iterrows():
            appearance_date = pd.to_datetime(row['game_date'])
            
            if appearance_date >= decline_start_date:
                # Calculate days since decline started
                days_since_decline = (appearance_date - decline_start_date).days
                
                # Apply progressive velocity decline
                if days_since_decline <= self.injury_patterns['velocity_decline']['initial_days']:
                    decline_range = self.injury_patterns['velocity_decline']['initial_decline']
                    decline_factor = np.random.uniform(decline_range[0], decline_range[1])
                elif days_since_decline <= self.injury_patterns['velocity_decline']['accelerated_days']:
                    decline_range = self.injury_patterns['velocity_decline']['accelerated_decline']
                    decline_factor = np.random.uniform(decline_range[0], decline_range[1])
                else:
                    decline_range = self.injury_patterns['velocity_decline']['severe_decline']
                    decline_factor = np.random.uniform(decline_range[0], decline_range[1])
                
                # Apply decline to velocity
                original_vel = row['avg_vel']
                new_vel = original_vel + decline_factor
                appearances.at[idx, 'avg_vel'] = max(88, new_vel)  # Don't go below 88 MPH


def validate_mock_data(data: Dict) -> Dict:
    """Validate generated mock data quality."""
    
    validation_results = {
        'pitches': validate_pitch_data(data['pitches']),
        'appearances': validate_appearance_data(data['appearances']),
        'injuries': validate_injury_data(data['injuries'])
    }
    
    return validation_results


def validate_pitch_data(pitches_df: pd.DataFrame) -> Dict:
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


def validate_appearance_data(appearances_df: pd.DataFrame) -> Dict:
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


def validate_injury_data(injuries: List[Dict]) -> Dict:
    """Validate injury data quality."""
    
    issues = []
    
    for injury in injuries:
        # Check date format
        try:
            pd.to_datetime(injury['il_start'])
            pd.to_datetime(injury['il_end'])
        except:
            issues.append(f"Invalid date format in injury {injury['pitcher_id']}")
        
        # Check that end date is after start date
        if pd.to_datetime(injury['il_end']) <= pd.to_datetime(injury['il_start']):
            issues.append(f"End date before start date in injury {injury['pitcher_id']}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'total_injuries': len(injuries)
    }


def export_mock_data(data: Dict, output_dir: str, format: str = 'csv') -> None:
    """Export mock data to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if format == 'csv':
        # Export to CSV files
        data['pitches'].to_csv(f'{output_dir}/pitches.csv', index=False)
        data['appearances'].to_csv(f'{output_dir}/appearances.csv', index=False)
        
        if data['injuries']:
            injuries_df = pd.DataFrame(data['injuries'])
            injuries_df.to_csv(f'{output_dir}/injuries.csv', index=False)
        
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
        
        if data['injuries']:
            injuries_df = pd.DataFrame(data['injuries'])
            injuries_df.to_parquet(f'{output_dir}/injuries.parquet', index=False)


def main():
    """Command line interface for mock data generation."""
    
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
