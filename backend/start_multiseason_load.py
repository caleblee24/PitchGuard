#!/usr/bin/env python3
"""
Start Multi-Season MLB Data Loading Process
This script initiates the comprehensive data loading for 2022-2024 seasons.
"""

import sys
import os
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiseason_data_loader import MultiSeasonDataLoader

def main():
    """Main function to start multi-season data loading."""
    print("🚀 Starting Multi-Season MLB Data Loading Process")
    print("=" * 60)
    print(f"📅 Target Seasons: 2022, 2023, 2024")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize the loader
    print("🔧 Initializing Multi-Season Data Loader...")
    loader = MultiSeasonDataLoader()
    
    # Setup database indices
    print("📊 Setting up database indices...")
    loader.setup_database_indices()
    
    # Check existing data
    print("🔍 Checking existing data...")
    existing_data = loader.check_existing_data()
    
    print(f"\n📈 Current Database State:")
    print(f"   - Total pitches: {existing_data['total_pitches']:,}")
    print(f"   - Total appearances: {existing_data['total_appearances']:,}")
    print(f"   - Total pitchers: {existing_data['total_pitchers']}")
    
    # Ask user if they want to proceed
    print(f"\n❓ Do you want to proceed with multi-season data loading?")
    print(f"   This will load 2022-2024 MLB data (may take several hours)")
    print(f"   Existing data will be preserved (idempotent loading)")
    
    response = input("   Continue? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("❌ Multi-season data loading cancelled.")
        return
    
    print(f"\n🎯 Starting multi-season data load...")
    print("   This process will:")
    print("   - Load pitches and appearances for 2022-2024")
    print("   - Process data in monthly batches")
    print("   - Skip existing data (idempotent)")
    print("   - Generate quality reports")
    print()
    
    try:
        # Start the full load
        results = loader.run_full_load([2022, 2023, 2024])
        
        print(f"\n🎉 Multi-Season Data Loading Complete!")
        print("=" * 60)
        print(f"📊 Final Results:")
        print(f"   - Total pitches loaded: {results['total_stats']['total_pitches']:,}")
        print(f"   - Total appearances loaded: {results['total_stats']['total_appearances']:,}")
        print(f"   - Unique pitchers: {results['total_stats']['total_pitchers']}")
        print(f"   - Errors encountered: {len(results['total_stats']['errors'])}")
        
        if results['total_stats']['errors']:
            print(f"\n⚠️  Errors encountered:")
            for error in results['total_stats']['errors'][:5]:
                print(f"   - {error}")
        
        print(f"\n📄 Quality report generated: docs/data_quality_gates.md")
        print(f"📄 Data ingest report: docs/data_ingest_report.md")
        
        print(f"\n✅ Next Steps:")
        print(f"   1. Review data quality report")
        print(f"   2. Proceed with injury label generation")
        print(f"   3. Run feature engineering")
        print(f"   4. Train enhanced model")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Multi-season data loading interrupted by user.")
        print(f"   Progress has been saved. You can resume later.")
        
    except Exception as e:
        print(f"\n❌ Multi-season data loading failed: {str(e)}")
        print(f"   Check logs for details.")
        raise

if __name__ == "__main__":
    main()

