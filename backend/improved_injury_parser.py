#!/usr/bin/env python3
"""
Improved Injury Parsing Algorithm
Uses advanced NLP techniques to accurately classify injuries
"""

import re
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedInjuryParser:
    """Advanced injury parsing with multiple classification strategies"""
    
    def __init__(self):
        self._init_classification_dictionaries()
        self._init_medical_terms()
        self._init_context_patterns()
    
    def _init_classification_dictionaries(self):
        """Initialize comprehensive classification dictionaries"""
        
        # Enhanced injury location mappings with medical terminology
        self.injury_locations = {
            'elbow': [
                'elbow', 'ulnar', 'flexor', 'extensor', 'ucl', 'tcl', 'medial', 'lateral', 
                'epicondyle', 'olecranon', 'radial', 'cubital', 'elbow joint', 'forearm',
                'throwing arm', 'pitching arm'
            ],
            'shoulder': [
                'shoulder', 'rotator cuff', 'labrum', 'ac joint', 'biceps', 'deltoid', 
                'supraspinatus', 'infraspinatus', 'teres minor', 'subscapularis', 
                'glenohumeral', 'scapula', 'clavicle', 'shoulder joint', 'rotator'
            ],
            'knee': [
                'knee', 'acl', 'mcl', 'meniscus', 'patella', 'ligament', 'tendon', 
                'cartilage', 'patellar', 'quadriceps tendon', 'knee joint', 'collateral',
                'cruciate', 'medial collateral', 'lateral collateral'
            ],
            'back': [
                'back', 'spine', 'lumbar', 'thoracic', 'disc', 'herniated', 'bulging', 
                'sciatica', 'vertebrae', 'spinal', 'lower back', 'upper back', 'mid back',
                'intervertebral', 'vertebral'
            ],
            'oblique': [
                'oblique', 'side', 'intercostal', 'rib', 'flank', 'abdominal', 'core',
                'side muscle', 'rib cage', 'intercostal muscle'
            ],
            'hamstring': [
                'hamstring', 'thigh', 'quadriceps', 'adductor', 'abductor', 'groin',
                'posterior thigh', 'biceps femoris', 'semitendinosus', 'semimembranosus',
                'leg muscle', 'upper leg'
            ],
            'forearm': [
                'forearm', 'wrist', 'hand', 'finger', 'thumb', 'carpal', 'tunnel',
                'flexor', 'extensor', 'pronator', 'supinator', 'wrist joint', 'hand joint'
            ],
            'ankle': [
                'ankle', 'foot', 'achilles', 'heel', 'plantar', 'fasciitis', 'ankle joint',
                'tibia', 'fibula', 'talus', 'calcaneus', 'ankle ligament', 'foot ligament'
            ],
            'hip': [
                'hip', 'groin', 'adductor', 'abductor', 'flexor', 'labrum', 'hip joint',
                'pelvis', 'femur', 'acetabulum', 'hip flexor', 'hip abductor'
            ],
            'neck': [
                'neck', 'cervical', 'whiplash', 'strain', 'cervical spine', 'neck muscle',
                'trapezius', 'sternocleidomastoid', 'cervical vertebrae'
            ],
            'calf': [
                'calf', 'gastrocnemius', 'soleus', 'achilles', 'lower leg', 'posterior leg',
                'calf muscle', 'gastroc', 'soleus muscle'
            ],
            'other': [
                'illness', 'covid', 'personal', 'bereavement', 'paternity', 'mental health',
                'concussion', 'head injury', 'general', 'undisclosed'
            ]
        }
        
        # Enhanced injury type mappings
        self.injury_types = {
            'strain': [
                'strain', 'pulled', 'muscle strain', 'muscle pull', 'overstretched',
                'muscle injury', 'muscle tear', 'grade 1', 'grade 2', 'grade 3'
            ],
            'sprain': [
                'sprain', 'twisted', 'ligament sprain', 'joint sprain', 'ligament injury',
                'joint injury', 'ligament tear', 'joint instability'
            ],
            'tear': [
                'tear', 'rupture', 'torn', 'complete tear', 'partial tear', 'full thickness',
                'tendon tear', 'ligament tear', 'muscle tear', 'labral tear'
            ],
            'fracture': [
                'fracture', 'broken', 'break', 'stress fracture', 'hairline', 'bone injury',
                'bone break', 'stress reaction', 'bone stress'
            ],
            'surgery': [
                'surgery', 'surgical', 'operation', 'procedure', 'reconstruction',
                'repair', 'arthroscopic', 'open surgery', 'surgical repair'
            ],
            'inflammation': [
                'inflammation', 'itis', 'tendonitis', 'bursitis', 'synovitis',
                'inflammatory', 'tendinitis', 'bursa', 'synovial'
            ],
            'soreness': [
                'soreness', 'tightness', 'discomfort', 'pain', 'tenderness',
                'muscle soreness', 'stiffness', 'ache', 'discomfort'
            ],
            'dislocation': [
                'dislocation', 'subluxation', 'popped out', 'joint dislocation',
                'shoulder dislocation', 'patellar dislocation'
            ],
            'contusion': [
                'contusion', 'bruise', 'hematoma', 'bruising', 'impact injury',
                'soft tissue injury'
            ],
            'concussion': [
                'concussion', 'head injury', 'head trauma', 'brain injury',
                'neurological', 'cognitive'
            ],
            'infection': [
                'infection', 'bacterial', 'viral', 'fungal', 'septic', 'infected',
                'bacterium', 'virus'
            ],
            'other': [
                'unknown', 'undisclosed', 'personal', 'illness', 'general',
                'non-specific', 'vague'
            ]
        }
        
        # Severity classification with more granular levels
        self.severity_keywords = {
            'severe': [
                'season', 'out for year', 'torn', 'rupture', 'surgery', '60-day', 
                'out indefinitely', 'career ending', 'major', 'significant',
                'complete tear', 'full thickness', 'grade 3', 'severe'
            ],
            'moderate': [
                'long-term', 'extended', 'weeks', 'month', '30-day', 'significant', 
                'major', 'partial tear', 'grade 2', 'moderate', 'substantial'
            ],
            'mild': [
                '10-day', 'day-to-day', 'soreness', 'tightness', 'questionable', 
                'probable', 'minor', 'grade 1', 'mild', 'slight', 'minimal'
            ]
        }
    
    def _init_medical_terms(self):
        """Initialize medical terminology mappings"""
        self.medical_terms = {
            'ulnar collateral ligament': 'elbow',
            'tommy john': 'elbow',
            'rotator cuff': 'shoulder',
            'labrum': 'shoulder',
            'acl': 'knee',
            'mcl': 'knee',
            'meniscus': 'knee',
            'herniated disc': 'back',
            'bulging disc': 'back',
            'sciatica': 'back',
            'plantar fasciitis': 'ankle',
            'achilles tendon': 'ankle',
            'carpal tunnel': 'forearm',
            'tennis elbow': 'elbow',
            'golfer elbow': 'elbow',
            'shin splints': 'ankle',
            'it band': 'knee',
            'quad strain': 'hamstring',
            'hamstring strain': 'hamstring',
            'oblique strain': 'oblique',
            'core injury': 'oblique'
        }
    
    def _init_context_patterns(self):
        """Initialize context-based patterns for inference"""
        self.context_patterns = {
            'pitching_related': [
                'throwing', 'pitching', 'arm', 'delivery', 'motion', 'windup',
                'follow through', 'release point'
            ],
            'running_related': [
                'running', 'sprinting', 'base running', 'leg', 'speed', 'acceleration'
            ],
            'fielding_related': [
                'fielding', 'diving', 'sliding', 'jumping', 'landing', 'impact'
            ],
            'weight_training': [
                'lifting', 'weight', 'strength', 'resistance', 'training', 'exercise'
            ]
        }
    
    def parse_injury_text(self, text: str) -> Dict[str, str]:
        """Parse injury text and return comprehensive classification"""
        text_lower = text.lower()
        
        # Initialize result
        result = {
            'injury_type': 'unknown',
            'injury_location': 'unknown',
            'severity': 'unknown',
            'confidence_score': 0.0,
            'parsing_method': 'none'
        }
        
        # Method 1: Direct keyword matching
        location = self._find_injury_location_direct(text_lower)
        injury_type = self._find_injury_type_direct(text_lower)
        severity = self._find_severity_direct(text_lower)
        
        if location != 'unknown' or injury_type != 'unknown' or severity != 'unknown':
            result.update({
                'injury_location': location,
                'injury_type': injury_type,
                'severity': severity,
                'confidence_score': 0.7,
                'parsing_method': 'direct_keyword'
            })
        
        # Method 2: Medical terminology matching
        if result['injury_location'] == 'unknown':
            medical_location = self._match_medical_terms(text_lower)
            if medical_location != 'unknown':
                result.update({
                    'injury_location': medical_location,
                    'confidence_score': result['confidence_score'] + 0.1,
                    'parsing_method': result['parsing_method'] + '+medical'
                })
        
        # Method 3: Context-based inference
        if result['injury_location'] == 'unknown':
            context_location = self._infer_from_context(text_lower)
            if context_location != 'unknown':
                result.update({
                    'injury_location': context_location,
                    'confidence_score': result['confidence_score'] + 0.05,
                    'parsing_method': result['parsing_method'] + '+context'
                })
        
        # Method 4: Pattern-based inference
        if result['injury_type'] == 'unknown':
            pattern_type = self._infer_from_patterns(text_lower)
            if pattern_type != 'unknown':
                result.update({
                    'injury_type': pattern_type,
                    'confidence_score': result['confidence_score'] + 0.05,
                    'parsing_method': result['parsing_method'] + '+patterns'
                })
        
        # Calculate final confidence score
        result['confidence_score'] = self._calculate_final_confidence(text, result)
        
        return result
    
    def _find_injury_location_direct(self, text: str) -> str:
        """Find injury location using direct keyword matching"""
        for location, keywords in self.injury_locations.items():
            if any(keyword in text for keyword in keywords):
                return location
        return "unknown"
    
    def _find_injury_type_direct(self, text: str) -> str:
        """Find injury type using direct keyword matching"""
        for injury_type, keywords in self.injury_types.items():
            if any(keyword in text for keyword in keywords):
                return injury_type
        return "unknown"
    
    def _find_severity_direct(self, text: str) -> str:
        """Find severity using direct keyword matching"""
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in text for keyword in keywords):
                return severity
        return "unknown"
    
    def _match_medical_terms(self, text: str) -> str:
        """Match specific medical terminology"""
        for medical_term, location in self.medical_terms.items():
            if medical_term in text:
                return location
        return "unknown"
    
    def _infer_from_context(self, text: str) -> str:
        """Infer injury location from context"""
        # Check for pitching-related context
        if any(word in text for word in self.context_patterns['pitching_related']):
            return "elbow"  # Most common for pitchers
        
        # Check for running-related context
        if any(word in text for word in self.context_patterns['running_related']):
            return "hamstring"  # Common running injury
        
        # Check for fielding-related context
        if any(word in text for word in self.context_patterns['fielding_related']):
            return "back"  # Common from diving/sliding
        
        return "unknown"
    
    def _infer_from_patterns(self, text: str) -> str:
        """Infer injury type from patterns"""
        # Look for surgery indicators
        if any(word in text for word in ['surgery', 'operation', 'procedure']):
            return "surgery"
        
        # Look for inflammation indicators
        if any(word in text for word in ['itis', 'inflammation', 'tendonitis']):
            return "inflammation"
        
        # Look for strain indicators
        if any(word in text for word in ['pulled', 'strain', 'overstretched']):
            return "strain"
        
        # Look for soreness indicators
        if any(word in text for word in ['soreness', 'tightness', 'discomfort']):
            return "soreness"
        
        return "unknown"
    
    def _calculate_final_confidence(self, original_text: str, result: Dict[str, str]) -> float:
        """Calculate final confidence score"""
        score = result['confidence_score']
        
        # Bonus for having both location and type
        if result['injury_location'] != 'unknown' and result['injury_type'] != 'unknown':
            score += 0.1
        
        # Bonus for having severity
        if result['severity'] != 'unknown':
            score += 0.05
        
        # Bonus for longer, more detailed text
        if len(original_text) > 50:
            score += 0.05
        
        # Bonus for specific medical terms
        if any(term in original_text.lower() for term in self.medical_terms.keys()):
            score += 0.1
        
        # Penalty for very short text
        if len(original_text) < 10:
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def update_injury_records(self):
        """Update existing injury records with improved parsing"""
        conn = sqlite3.connect("pitchguard.db")
        cursor = conn.cursor()
        
        # Get all injury records
        cursor.execute('''
            SELECT id, notes, raw_text 
            FROM injury_records 
            WHERE injury_type = 'unknown' OR injury_location = 'unknown'
        ''')
        
        records = cursor.fetchall()
        logger.info(f"Found {len(records)} records to update")
        
        updated_count = 0
        for record_id, notes, raw_text in records:
            try:
                # Combine notes and raw_text for parsing
                text_to_parse = f"{notes} {raw_text}".strip()
                
                if text_to_parse:
                    # Parse with improved algorithm
                    parsed_result = self.parse_injury_text(text_to_parse)
                    
                    # Update database if we got better results
                    if (parsed_result['injury_type'] != 'unknown' or 
                        parsed_result['injury_location'] != 'unknown'):
                        
                        cursor.execute('''
                            UPDATE injury_records 
                            SET injury_type = ?, injury_location = ?, severity = ?,
                                confidence_score = ?, data_quality = 'improved_parsing'
                            WHERE id = ?
                        ''', (
                            parsed_result['injury_type'],
                            parsed_result['injury_location'],
                            parsed_result['severity'],
                            parsed_result['confidence_score'],
                            record_id
                        ))
                        updated_count += 1
                        
            except Exception as e:
                logger.warning(f"Error updating record {record_id}: {e}")
                continue
        
        conn.commit()
        conn.close()
        logger.info(f"Updated {updated_count} injury records with improved parsing")
    
    def analyze_parsing_improvements(self) -> Dict:
        """Analyze the improvements from the enhanced parsing"""
        conn = sqlite3.connect("pitchguard.db")
        cursor = conn.cursor()
        
        # Get parsing statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN injury_type != 'unknown' THEN 1 ELSE 0 END) as classified_type,
                SUM(CASE WHEN injury_location != 'unknown' THEN 1 ELSE 0 END) as classified_location,
                SUM(CASE WHEN severity != 'unknown' THEN 1 ELSE 0 END) as classified_severity,
                AVG(confidence_score) as avg_confidence
            FROM injury_records
        ''')
        
        stats = cursor.fetchone()
        
        # Get breakdown by injury type
        cursor.execute('''
            SELECT injury_type, COUNT(*) 
            FROM injury_records 
            WHERE injury_type != 'unknown'
            GROUP BY injury_type 
            ORDER BY COUNT(*) DESC
        ''')
        type_breakdown = dict(cursor.fetchall())
        
        # Get breakdown by injury location
        cursor.execute('''
            SELECT injury_location, COUNT(*) 
            FROM injury_records 
            WHERE injury_location != 'unknown'
            GROUP BY injury_location 
            ORDER BY COUNT(*) DESC
        ''')
        location_breakdown = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_records': stats[0],
            'classified_type': stats[1],
            'classified_location': stats[2],
            'classified_severity': stats[3],
            'avg_confidence': stats[4],
            'type_breakdown': type_breakdown,
            'location_breakdown': location_breakdown,
            'type_classification_rate': stats[1] / stats[0] if stats[0] > 0 else 0,
            'location_classification_rate': stats[2] / stats[0] if stats[0] > 0 else 0
        }

def main():
    """Main execution function"""
    parser = ImprovedInjuryParser()
    
    # Update existing records
    parser.update_injury_records()
    
    # Analyze improvements
    analysis = parser.analyze_parsing_improvements()
    
    print("\n" + "="*60)
    print("IMPROVED INJURY PARSING ANALYSIS")
    print("="*60)
    print(f"Total Records: {analysis['total_records']}")
    print(f"Type Classification Rate: {analysis['type_classification_rate']:.1%}")
    print(f"Location Classification Rate: {analysis['location_classification_rate']:.1%}")
    print(f"Average Confidence Score: {analysis['avg_confidence']:.2f}")
    print(f"\nInjury Type Breakdown: {analysis['type_breakdown']}")
    print(f"\nInjury Location Breakdown: {analysis['location_breakdown']}")

if __name__ == "__main__":
    main()
