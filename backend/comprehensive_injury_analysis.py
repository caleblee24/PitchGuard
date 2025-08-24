#!/usr/bin/env python3
"""
Comprehensive Injury Data Analysis
Analyze all collected injury data and parsing improvements
"""

import sqlite3
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_injury_data():
    """Comprehensive analysis of injury data"""
    conn = sqlite3.connect("pitchguard.db")
    
    # Get all injury records
    query = "SELECT * FROM injury_records ORDER BY created_at DESC"
    df = pd.read_sql_query(query, conn)
    
    print("="*80)
    print("COMPREHENSIVE INJURY DATA ANALYSIS")
    print("="*80)
    
    print(f"\n📊 TOTAL INJURY RECORDS: {len(df)}")
    
    if len(df) > 0:
        # Basic statistics
        print(f"\n📈 BASIC STATISTICS:")
        print(f"  Date Range: {df['created_at'].min()} to {df['created_at'].max()}")
        print(f"  Unique Pitchers: {df['pitcher_name'].nunique()}")
        print(f"  Average Confidence Score: {df['confidence_score'].mean():.2f}")
        
        # Source analysis
        print(f"\n📰 SOURCE ANALYSIS:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {source}: {count} records ({percentage:.1f}%)")
        
        # Injury type analysis
        print(f"\n🏥 INJURY TYPE ANALYSIS:")
        type_counts = df['injury_type'].value_counts()
        for injury_type, count in type_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {injury_type}: {count} records ({percentage:.1f}%)")
        
        # Injury location analysis
        print(f"\n📍 INJURY LOCATION ANALYSIS:")
        location_counts = df['injury_location'].value_counts()
        for location, count in location_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {location}: {count} records ({percentage:.1f}%)")
        
        # Severity analysis
        print(f"\n⚠️ SEVERITY ANALYSIS:")
        severity_counts = df['severity'].value_counts()
        for severity, count in severity_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {severity}: {count} records ({percentage:.1f}%)")
        
        # Data quality analysis
        print(f"\n🔍 DATA QUALITY ANALYSIS:")
        quality_counts = df['data_quality'].value_counts()
        for quality, count in quality_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {quality}: {count} records ({percentage:.1f}%)")
        
        # Confidence score distribution
        print(f"\n📊 CONFIDENCE SCORE DISTRIBUTION:")
        confidence_ranges = [
            (0.0, 0.3, "Low"),
            (0.3, 0.6, "Medium"),
            (0.6, 0.8, "High"),
            (0.8, 1.0, "Very High")
        ]
        
        for min_score, max_score, label in confidence_ranges:
            count = len(df[(df['confidence_score'] >= min_score) & (df['confidence_score'] < max_score)])
            percentage = (count / len(df)) * 100
            print(f"  {label} ({min_score}-{max_score}): {count} records ({percentage:.1f}%)")
        
        # Sample of high-confidence records
        print(f"\n✅ SAMPLE HIGH-CONFIDENCE RECORDS:")
        high_confidence = df[df['confidence_score'] >= 0.6].head(10)
        if len(high_confidence) > 0:
            for _, record in high_confidence.iterrows():
                print(f"  {record['pitcher_name']}: {record['injury_type']} - {record['injury_location']} ({record['severity']}) - Confidence: {record['confidence_score']:.2f}")
        else:
            print("  No high-confidence records found")
        
        # Sample of records needing improvement
        print(f"\n🔧 SAMPLE RECORDS NEEDING IMPROVEMENT:")
        low_confidence = df[df['confidence_score'] < 0.3].head(10)
        if len(low_confidence) > 0:
            for _, record in low_confidence.iterrows():
                print(f"  {record['pitcher_name']}: {record['injury_type']} - {record['injury_location']} - Notes: {record['notes'][:50]}...")
        else:
            print("  No low-confidence records found")
        
        # Parsing improvement analysis
        print(f"\n🚀 PARSING IMPROVEMENT ANALYSIS:")
        improved_records = df[df['data_quality'] == 'improved_parsing']
        print(f"  Records with improved parsing: {len(improved_records)}")
        
        if len(improved_records) > 0:
            print(f"  Average confidence of improved records: {improved_records['confidence_score'].mean():.2f}")
            
            improved_type_counts = improved_records['injury_type'].value_counts()
            print(f"  Injury types in improved records:")
            for injury_type, count in improved_type_counts.items():
                if injury_type != 'unknown':
                    print(f"    {injury_type}: {count}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        # Calculate classification rates
        type_classification_rate = (len(df[df['injury_type'] != 'unknown']) / len(df)) * 100
        location_classification_rate = (len(df[df['injury_location'] != 'unknown']) / len(df)) * 100
        severity_classification_rate = (len(df[df['severity'] != 'unknown']) / len(df)) * 100
        
        print(f"  Injury Type Classification Rate: {type_classification_rate:.1f}%")
        print(f"  Injury Location Classification Rate: {location_classification_rate:.1f}%")
        print(f"  Severity Classification Rate: {severity_classification_rate:.1f}%")
        
        if type_classification_rate < 50:
            print(f"  ⚠️ Need to improve injury type classification")
        
        if location_classification_rate < 50:
            print(f"  ⚠️ Need to improve injury location classification")
        
        if severity_classification_rate < 50:
            print(f"  ⚠️ Need to improve severity classification")
        
        # Data source recommendations
        if len(df) < 500:
            print(f"  📈 Need more injury data - target: 500+ records")
        
        # Quality recommendations
        avg_confidence = df['confidence_score'].mean()
        if avg_confidence < 0.6:
            print(f"  🔧 Need to improve parsing algorithms - current avg confidence: {avg_confidence:.2f}")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"  1. Collect more injury data from additional sources")
        print(f"  2. Improve parsing algorithms for better classification")
        print(f"  3. Add more medical terminology to parsing dictionaries")
        print(f"  4. Implement machine learning-based classification")
        print(f"  5. Validate data quality with medical experts")
    
    conn.close()

def create_visualizations():
    """Create visualizations of the injury data"""
    conn = sqlite3.connect("pitchguard.db")
    df = pd.read_sql_query("SELECT * FROM injury_records", conn)
    conn.close()
    
    if len(df) == 0:
        print("No data to visualize")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PitchGuard Injury Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Source distribution
    source_counts = df['source'].value_counts()
    axes[0, 0].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Data Sources')
    
    # 2. Injury location distribution (excluding unknown)
    location_counts = df[df['injury_location'] != 'unknown']['injury_location'].value_counts()
    if len(location_counts) > 0:
        axes[0, 1].bar(location_counts.index, location_counts.values)
        axes[0, 1].set_title('Injury Locations (Classified)')
        axes[0, 1].tick_params(axis='x', rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, 'No classified locations', ha='center', va='center')
        axes[0, 1].set_title('Injury Locations')
    
    # 3. Severity distribution
    severity_counts = df['severity'].value_counts()
    axes[1, 0].bar(severity_counts.index, severity_counts.values)
    axes[1, 0].set_title('Injury Severity')
    
    # 4. Confidence score distribution
    axes[1, 1].hist(df['confidence_score'], bins=20, alpha=0.7)
    axes[1, 1].set_title('Confidence Score Distribution')
    axes[1, 1].set_xlabel('Confidence Score')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('injury_data_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'injury_data_analysis.png'")

if __name__ == "__main__":
    analyze_injury_data()
    try:
        create_visualizations()
    except Exception as e:
        print(f"Could not create visualizations: {e}")
