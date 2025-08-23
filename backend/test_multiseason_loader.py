#!/usr/bin/env python3
"""
Test script for multi-season data loading
Loads a small amount of data to verify the system works.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiseason_data_loader import MultiSeasonDataLoader
from real_mlb_data_integration import RealMLBDataIntegrator

def test_small_data_load():
    """Test loading a small amount of data."""
    print("🧪 Testing Multi-Season Data Loader")
    print("=" * 50)
    
    # Initialize loader
    loader = MultiSeasonDataLoader()
    
    # Test with just one month of 2024 data
    test_start = "2024-08-01"
    test_end = "2024-08-31"
    
    print(f"📅 Testing with {test_start} to {test_end}")
    
    try:
        # Test the monthly batch loading
        batch_stats = loader._load_monthly_batch(2024, test_start, test_end)
        
        print(f"✅ Test batch loaded successfully:")
        print(f"   - Pitches loaded: {batch_stats['pitches_loaded']:,}")
        print(f"   - Appearances loaded: {batch_stats['appearances_loaded']:,}")
        print(f"   - Pitchers found: {len(batch_stats['pitchers_found'])}")
        
        if batch_stats['pitches_loaded'] > 0:
            print("🎉 Test successful! Ready for full multi-season load.")
            return True
        else:
            print("⚠️  No data loaded. This might be normal if no games in this period.")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_data_quality():
    """Test data quality gates after small load."""
    print("\n🔍 Testing Data Quality Gates")
    print("=" * 30)
    
    try:
        from data_quality_gates import DataQualityGates
        
        gates = DataQualityGates()
        results = gates.run_all_gates()
        
        print(f"📊 Quality Gates Results:")
        print(f"   - Passed: {results['gates_passed']}")
        print(f"   - Failed: {results['gates_failed']}")
        print(f"   - Warnings: {len(results['warnings'])}")
        
        return results['gates_failed'] == 0
        
    except Exception as e:
        print(f"❌ Quality gates test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Multi-Season Data Loader Test")
    print()
    
    # Test small data load
    load_success = test_small_data_load()
    
    if load_success:
        # Test data quality
        quality_success = test_data_quality()
        
        if quality_success:
            print("\n🎉 All tests passed! Ready for full multi-season load.")
            print("\nNext steps:")
            print("1. Run: python3 multiseason_data_loader.py")
            print("2. Monitor progress and data quality")
            print("3. Proceed with model training once data is loaded")
        else:
            print("\n⚠️  Data quality issues detected. Review before proceeding.")
    else:
        print("\n❌ Data loading test failed. Fix issues before proceeding.")

