# Operations Runbooks

## Overview

This document provides step-by-step procedures for running PitchGuard operations, including data processing, model training, API serving, and demo execution.

## Prerequisites

### Environment Setup

1. **Python Environment**
   ```bash
   # Create virtual environment
   python -m venv pitchguard_env
   source pitchguard_env/bin/activate  # On Windows: pitchguard_env\Scripts\activate
   
   # Install dependencies
   pip install -r backend/requirements.txt
   ```

2. **Database Setup**
   ```bash
   # For development (SQLite)
   # No setup required - will be created automatically
   
   # For production (PostgreSQL)
   createdb pitchguard
   psql pitchguard -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
   ```

3. **Environment Variables**
   ```bash
   # Copy example environment file
   cp backend/.env.example backend/.env
   
   # Edit environment variables
   nano backend/.env
   ```

## Data Operations

### 1. Mock Data Generation

**Purpose**: Generate realistic mock data for development and testing

**Procedure**:
```bash
cd backend

# Generate mock data
python -m etl.mock_data_generator \
  --output-dir ./data/mock \
  --num-pitchers 3 \
  --start-date 2024-04-01 \
  --end-date 2024-06-30 \
  --injury-scenario true
```

**Expected Output**:
- `./data/mock/pitches.csv` - Raw pitch data
- `./data/mock/appearances.csv` - Aggregated appearances
- `./data/mock/injuries.csv` - Injury data

**Verification**:
```bash
# Check data quality
python -m etl.data_validator --input-dir ./data/mock

# Expected output:
# ✓ 3 pitchers generated
# ✓ 90 days of data
# ✓ 1 injury scenario created
# ✓ All data constraints satisfied
```

### 2. ETL Pipeline Execution

**Purpose**: Process raw data into ML-ready features

**Procedure**:
```bash
cd backend

# Run complete ETL pipeline
python -m etl.pipeline \
  --data-source mock \
  --start-date 2024-04-01 \
  --end-date 2024-06-30 \
  --output-dir ./data/processed
```

**Pipeline Stages**:
1. **Ingestion**: Load and validate raw data
2. **Aggregation**: Convert pitches to appearances
3. **Feature Engineering**: Compute rolling features
4. **Labeling**: Generate injury labels

**Monitoring**:
```bash
# Check pipeline logs
tail -f logs/etl.log

# Expected log entries:
# INFO: Starting ETL pipeline for 2024-04-01 to 2024-06-30
# INFO: Ingested 2,700 pitch records
# INFO: Generated 90 appearance records
# INFO: Computed features for 270 snapshots
# INFO: Generated 15 positive injury labels
# INFO: ETL pipeline completed successfully
```

**Troubleshooting**:
```bash
# Check data quality issues
python -m etl.data_validator --input-dir ./data/processed --verbose

# Fix common issues
python -m etl.data_cleaner --input-dir ./data/processed --fix-missing
```

### 3. Database Operations

**Purpose**: Load processed data into database

**Procedure**:
```bash
cd backend

# Load data into database
python -m db.loader \
  --input-dir ./data/processed \
  --database-url $PG_URL
```

**Verification**:
```bash
# Check database contents
python -m db.validator --database-url $PG_URL

# Expected output:
# ✓ 270 feature snapshots loaded
# ✓ 90 appearances loaded
# ✓ 15 injuries loaded
# ✓ All constraints satisfied
```

## Model Operations

### 1. Model Training

**Purpose**: Train injury risk prediction model

**Procedure**:
```bash
cd backend

# Train model with default configuration
python -m modeling.train \
  --config ./config/model_config.yaml \
  --output-dir ./models \
  --model-name logistic_regression_v1
```

**Training Configuration** (`config/model_config.yaml`):
```yaml
model:
  type: logistic_regression
  class_weight: balanced
  max_iter: 1000

features:
  selection_method: f_classif
  num_features: 10
  scaling: standard

training:
  train_end_date: 2024-05-31
  val_start_date: 2024-06-01
  cv_folds: 5

evaluation:
  metrics:
    - auc
    - precision_at_top_10
    - recall_at_top_10
    - brier_score
    - calibration_error
```

**Expected Output**:
```
Training started...
✓ Feature selection completed (10 features selected)
✓ Model training completed
✓ Cross-validation completed
✓ Model evaluation completed

Performance Metrics:
- AUC: 0.78
- Precision@Top-10%: 0.65
- Recall@Top-10%: 0.70
- Brier Score: 0.12
- Calibration Error: 0.03

Model saved to: ./models/logistic_regression_v1.pkl
```

**Verification**:
```bash
# Validate model performance
python -m modeling.evaluate \
  --model-path ./models/logistic_regression_v1.pkl \
  --test-data ./data/processed/features_test.csv

# Expected output:
# ✓ All performance thresholds met
# ✓ Model ready for deployment
```

### 2. Model Deployment

**Purpose**: Deploy trained model for API serving

**Procedure**:
```bash
cd backend

# Deploy model to production
python -m modeling.deploy \
  --model-path ./models/logistic_regression_v1.pkl \
  --deploy-name production \
  --activate true
```

**Verification**:
```bash
# Check deployed model
python -m modeling.status

# Expected output:
# Active Model: logistic_regression_v1
# Version: 1.0.0
# Deployed: 2024-04-15 10:30:00
# Status: Active
```

### 3. Model Monitoring

**Purpose**: Monitor model performance in production

**Procedure**:
```bash
cd backend

# Check model performance
python -m modeling.monitor \
  --model-name logistic_regression_v1 \
  --days 7

# Expected output:
# Model Performance (Last 7 days):
# - Predictions: 1,250
# - AUC: 0.76
# - Calibration Error: 0.04
# - Data Drift: None detected
```

## API Operations

### 1. API Server Startup

**Purpose**: Start the FastAPI server for risk assessment

**Procedure**:
```bash
cd backend

# Start development server
python -m uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info
```

**Expected Output**:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Verification**:
```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health

# Expected response:
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "services": {
      "database": "healthy",
      "model_service": "healthy"
    }
  }
}
```

### 2. API Testing

**Purpose**: Test API endpoints with sample data

**Procedure**:
```bash
# Test risk assessment endpoint
curl -X POST http://localhost:8000/api/v1/risk/pitcher \
  -H "Content-Type: application/json" \
  -d '{
    "pitcher_id": 12345,
    "as_of_date": "2024-06-15"
  }'

# Expected response:
{
  "success": true,
  "data": {
    "pitcher_id": 12345,
    "risk_score": 0.23,
    "risk_percentage": 23,
    "risk_level": "medium",
    "top_contributors": [...]
  }
}

# Test workload history endpoint
curl "http://localhost:8000/api/v1/workload/pitcher/12345?start_date=2024-06-01&end_date=2024-06-15"

# Test pitchers list endpoint
curl "http://localhost:8000/api/v1/pitchers?team=NYY&limit=5"
```

### 3. API Performance Testing

**Purpose**: Test API performance under load

**Procedure**:
```bash
cd backend

# Run load test
python -m api.load_test \
  --endpoint /api/v1/risk/pitcher \
  --requests 100 \
  --concurrent 10 \
  --duration 60
```

**Expected Output**:
```
Load Test Results:
- Total Requests: 100
- Successful: 100
- Failed: 0
- Average Response Time: 45ms
- 95th Percentile: 78ms
- Requests/Second: 22.2
```

## Frontend Operations

### 1. Frontend Development Server

**Purpose**: Start React development server

**Procedure**:
```bash
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

**Expected Output**:
```
Compiled successfully!

You can now view pitchguard in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.1.100:3000

Note that the development build is not optimized.
To create a production build, use npm run build.
```

### 2. Frontend Testing

**Purpose**: Test frontend functionality

**Procedure**:
```bash
cd frontend

# Run unit tests
npm test

# Run integration tests
npm run test:integration

# Run end-to-end tests
npm run test:e2e
```

**Expected Output**:
```
Test Results:
✓ All unit tests passed (45/45)
✓ All integration tests passed (12/12)
✓ All E2E tests passed (8/8)
```

### 3. Frontend Build

**Purpose**: Create production build

**Procedure**:
```bash
cd frontend

# Create production build
npm run build

# Expected output:
# ✓ Build completed successfully
# ✓ Bundle size: 2.1MB
# ✓ Assets optimized
```

## Demo Operations

### 1. Demo Setup

**Purpose**: Prepare system for demonstration

**Procedure**:
```bash
# 1. Generate demo data
cd backend
python -m etl.mock_data_generator \
  --output-dir ./data/demo \
  --num-pitchers 5 \
  --start-date 2024-04-01 \
  --end-date 2024-06-30 \
  --injury-scenario true \
  --demo-mode true

# 2. Run ETL pipeline
python -m etl.pipeline \
  --data-source mock \
  --input-dir ./data/demo \
  --output-dir ./data/demo_processed

# 3. Load into database
python -m db.loader \
  --input-dir ./data/demo_processed \
  --database-url $PG_URL

# 4. Train demo model
python -m modeling.train \
  --config ./config/demo_config.yaml \
  --output-dir ./models \
  --model-name demo_model

# 5. Deploy demo model
python -m modeling.deploy \
  --model-path ./models/demo_model.pkl \
  --deploy-name demo \
  --activate true
```

### 2. Demo Execution

**Purpose**: Run live demonstration

**Procedure**:
```bash
# Terminal 1: Start backend
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend
cd frontend
npm start

# Terminal 3: Monitor logs
tail -f backend/logs/api.log
```

**Demo Flow**:
1. **Overview Dashboard**: Show pitcher list with risk indicators
2. **High-Risk Pitcher**: Click on pitcher with elevated risk
3. **Detail Analysis**: Show "why now" explanations
4. **Workload Trends**: Display velocity and pitch count trends
5. **Mitigation Suggestions**: Show actionable recommendations

### 3. Demo Verification

**Purpose**: Verify demo functionality

**Procedure**:
```bash
# Test demo endpoints
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/pitchers?limit=5
curl -X POST http://localhost:8000/api/v1/risk/pitcher \
  -H "Content-Type: application/json" \
  -d '{"pitcher_id": 12345, "as_of_date": "2024-06-15"}'

# Check frontend
open http://localhost:3000
```

## Monitoring and Maintenance

### 1. System Health Check

**Purpose**: Monitor overall system health

**Procedure**:
```bash
cd backend

# Run health check
python -m ops.health_check

# Expected output:
# ✓ Database connection: Healthy
# ✓ Model service: Healthy
# ✓ API endpoints: Healthy
# ✓ Data freshness: 2 hours ago
# ✓ System uptime: 7 days, 3 hours
```

### 2. Log Monitoring

**Purpose**: Monitor system logs for issues

**Procedure**:
```bash
# Monitor API logs
tail -f backend/logs/api.log | grep -E "(ERROR|WARNING)"

# Monitor ETL logs
tail -f backend/logs/etl.log | grep -E "(ERROR|WARNING)"

# Monitor model logs
tail -f backend/logs/model.log | grep -E "(ERROR|WARNING)"
```

### 3. Performance Monitoring

**Purpose**: Monitor system performance

**Procedure**:
```bash
# Check API performance
python -m ops.performance_monitor --service api --hours 24

# Check database performance
python -m ops.performance_monitor --service database --hours 24

# Check model performance
python -m ops.performance_monitor --service model --hours 24
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```bash
   # Check database status
   python -m db.status
   
   # Restart database connection
   python -m db.reconnect
   ```

2. **Model Loading Issues**
   ```bash
   # Check model status
   python -m modeling.status
   
   # Reload model
   python -m modeling.reload --model-name production
   ```

3. **API Response Issues**
   ```bash
   # Check API health
   curl http://localhost:8000/api/v1/health
   
   # Check API logs
   tail -f backend/logs/api.log
   ```

4. **Frontend Issues**
   ```bash
   # Clear cache and restart
   cd frontend
   npm run clean
   npm install
   npm start
   ```

### Emergency Procedures

1. **System Downtime**
   ```bash
   # Stop all services
   pkill -f "uvicorn"
   pkill -f "npm start"
   
   # Restart services
   cd backend && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
   cd frontend && npm start &
   ```

2. **Data Corruption**
   ```bash
   # Restore from backup
   python -m db.restore --backup-file ./backups/latest.sql
   
   # Regenerate features
   python -m etl.pipeline --force-regenerate
   ```

3. **Model Degradation**
   ```bash
   # Rollback to previous model
   python -m modeling.rollback --version previous
   
   # Retrain model
   python -m modeling.train --force-retrain
   ```

## Backup and Recovery

### 1. Database Backup

**Purpose**: Create database backup

**Procedure**:
```bash
cd backend

# Create backup
python -m db.backup --output-file ./backups/backup_$(date +%Y%m%d_%H%M%S).sql

# Verify backup
python -m db.verify_backup --backup-file ./backups/latest.sql
```

### 2. Model Backup

**Purpose**: Backup trained models

**Procedure**:
```bash
cd backend

# Backup models
python -m modeling.backup --output-dir ./backups/models

# Verify backup
python -m modeling.verify_backup --backup-dir ./backups/models
```

### 3. Configuration Backup

**Purpose**: Backup configuration files

**Procedure**:
```bash
# Backup configuration
tar -czf ./backups/config_$(date +%Y%m%d_%H%M%S).tar.gz \
  backend/.env \
  backend/config/ \
  frontend/.env
```
