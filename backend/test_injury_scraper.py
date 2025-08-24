#!/usr/bin/env python3
"""
Test script for injury data scraper
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_injury_scraper import AdvancedInjuryScraper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_scraper():
    """Test the injury scraper with a small sample"""
    print("Testing Advanced Injury Scraper...")
    
    scraper = AdvancedInjuryScraper()
    
    # Test individual sources first
    print("\n1. Testing ESPN API...")
    try:
        espn_injuries = scraper.scrape_espn_api_injuries()
        print(f"ESPN: {len(espn_injuries)} injuries found")
        if espn_injuries:
            print(f"Sample: {espn_injuries[0].pitcher_name} - {espn_injuries[0].injury_type}")
    except Exception as e:
        print(f"ESPN Error: {e}")
    
    print("\n2. Testing FantasyPros...")
    try:
        fp_injuries = scraper.scrape_fantasy_pros_injuries()
        print(f"FantasyPros: {len(fp_injuries)} injuries found")
        if fp_injuries:
            print(f"Sample: {fp_injuries[0].pitcher_name} - {fp_injuries[0].injury_type}")
    except Exception as e:
        print(f"FantasyPros Error: {e}")
    
    # Test comprehensive scraping
    print("\n3. Running comprehensive scrape...")
    try:
        all_injuries = scraper.run_comprehensive_scrape()
        print(f"Total injuries collected: {len(all_injuries)}")
        
        # Get summary
        summary = scraper.get_injury_summary()
        print(f"\nSummary:")
        print(f"Total in DB: {summary['total_injuries']}")
        print(f"By source: {summary['by_source']}")
        print(f"By type: {summary['by_type']}")
        print(f"By location: {summary['by_location']}")
        
    except Exception as e:
        print(f"Comprehensive scrape error: {e}")

if __name__ == "__main__":
    test_scraper()
