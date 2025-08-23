#!/usr/bin/env python3
"""
Simple API Test Script
Tests the PitchGuard API endpoints with existing data.
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000/api/v1"

def test_pitchers_endpoint():
    """Test the pitchers endpoint."""
    print("🧪 Testing /pitchers endpoint...")
    
    response = requests.get(f"{BASE_URL}/pitchers")
    if response.status_code == 200:
        pitchers = response.json()
        print(f"✅ Success! Found {len(pitchers)} pitchers")
        print(f"   Sample pitcher: {pitchers[0]}")
        return pitchers
    else:
        print(f"❌ Failed: {response.status_code}")
        return None

def test_risk_assessment(pitcher_id, as_of_date):
    """Test the risk assessment endpoint."""
    print(f"🧪 Testing risk assessment for pitcher {pitcher_id} on {as_of_date}...")
    
    payload = {
        "pitcher_id": pitcher_id,
        "as_of_date": as_of_date
    }
    
    response = requests.post(f"{BASE_URL}/risk/pitcher", json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success!")
        print(f"   Risk Level: {result.get('risk_level')}")
        print(f"   Risk Score: {result.get('risk_score'):.3f}")
        print(f"   Confidence: {result.get('confidence')}")
        print(f"   Data Completeness: {result.get('data_completeness')}")
        if result.get('risk_factors'):
            print(f"   Top Risk Factor: {result['risk_factors'][0]}")
        return result
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def test_workload_endpoint(pitcher_id, start_date, end_date):
    """Test the workload endpoint."""
    print(f"🧪 Testing workload for pitcher {pitcher_id} from {start_date} to {end_date}...")
    
    params = {
        "start": start_date,
        "end": end_date
    }
    
    response = requests.get(f"{BASE_URL}/workload/pitcher/{pitcher_id}", params=params)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success!")
        print(f"   Data Points: {len(result.get('data_points', []))}")
        print(f"   Total Pitches: {result.get('total_pitches')}")
        print(f"   Avg Velocity: {result.get('avg_velocity')}")
        return result
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def test_health_endpoints():
    """Test health check endpoints."""
    print("🧪 Testing health endpoints...")
    
    # Test /health
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        print("✅ /health endpoint working")
    else:
        print(f"❌ /health failed: {response.status_code}")
    
    # Test /readiness
    response = requests.get(f"{BASE_URL}/readiness")
    if response.status_code == 200:
        print("✅ /readiness endpoint working")
    else:
        print(f"❌ /readiness failed: {response.status_code}")

def main():
    """Run all API tests."""
    print("🚀 Starting PitchGuard API Tests")
    print("=" * 50)
    
    # Test health endpoints
    test_health_endpoints()
    print()
    
    # Test pitchers endpoint
    pitchers = test_pitchers_endpoint()
    print()
    
    if pitchers:
        # Test risk assessment with first pitcher
        pitcher_id = pitchers[0]['pitcher_id']
        as_of_date = "2024-08-15"
        
        test_risk_assessment(pitcher_id, as_of_date)
        print()
        
        # Test workload endpoint
        start_date = "2024-07-01"
        end_date = "2024-08-01"
        
        test_workload_endpoint(pitcher_id, start_date, end_date)
        print()
        
        # Test multiple risk assessments
        print("🧪 Testing multiple risk assessments...")
        test_dates = ["2024-08-01", "2024-08-15", "2024-08-20"]
        for date in test_dates:
            test_risk_assessment(pitcher_id, date)
            print()
    
    print("🎉 API Testing Complete!")

if __name__ == "__main__":
    main()
