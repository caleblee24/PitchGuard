# Data Quality Gates Report

**Generated:** 2025-08-21T01:57:23.255787

## Summary

- **Gates Passed:** 3
- **Gates Failed:** 4
- **Warnings:** 1
- **Errors:** 4

## Errors

- Season 2024: Only 1,520 appearances (min: 5,000)
- Velocity too low: 43.8 MPH (min: 60.0 MPH); Spin rate too low: 0 RPM (min: 500 RPM)
- Too many duplicates: 41.77% (50,048 rows)
- Gate gate_monotonic_sanity failed with exception: no such function: CORR

## Warnings

- Found 50048 groups of duplicate pitches

## Detailed Results

### Row Counts & Season Distribution

- **Status:** ❌ FAILED
- **Message:** Season 2024: Only 1,520 appearances (min: 5,000)

### Missing Data Validation

- **Status:** ✅ PASSED
- **Message:** Missing data within acceptable limits

### Data Range Validation

- **Status:** ❌ FAILED
- **Message:** Velocity too low: 43.8 MPH (min: 60.0 MPH); Spin rate too low: 0 RPM (min: 500 RPM)

### Duplicate Detection

- **Status:** ❌ FAILED
- **Message:** Too many duplicates: 41.77% (50,048 rows)
- **Warnings:**
  - Found 50048 groups of duplicate pitches

### Temporal Consistency

- **Status:** ✅ PASSED
- **Message:** Temporal consistency validated

### Pitcher Consistency

- **Status:** ✅ PASSED
- **Message:** Pitcher consistency validated

