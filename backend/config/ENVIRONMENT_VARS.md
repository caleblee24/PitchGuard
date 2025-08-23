# Environment Variables Configuration

## Overview

This document defines all environment variables required for the PitchGuard backend system. These variables control database connections, API settings, model configuration, and operational parameters.

## Database Configuration

### Primary Database (PostgreSQL)

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `PG_URL` | PostgreSQL connection string | None | Yes | `postgresql://user:pass@localhost:5432/pitchguard` |
| `PG_HOST` | Database host | `localhost` | No | `db.pitchguard.com` |
| `PG_PORT` | Database port | `5432` | No | `5432` |
| `PG_DATABASE` | Database name | `pitchguard` | No | `pitchguard_prod` |
| `PG_USER` | Database username | None | No | `pitchguard_user` |
| `PG_PASSWORD` | Database password | None | No | `secure_password_123` |
| `PG_POOL_SIZE` | Connection pool size | `10` | No | `20` |
| `PG_MAX_OVERFLOW` | Max overflow connections | `20` | No | `30` |

### SQLite (Development Only)

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `SQLITE_PATH` | SQLite database file path | `./data/pitchguard.db` | No | `/app/data/pitchguard.db` |

## API Configuration

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `API_HOST` | API server host | `0.0.0.0` | No | `localhost` |
| `API_PORT` | API server port | `8000` | No | `8000` |
| `API_WORKERS` | Number of worker processes | `1` | No | `4` |
| `API_RELOAD` | Enable auto-reload (dev) | `false` | No | `true` |
| `CORS_ORIGINS` | Allowed CORS origins | `["http://localhost:3000"]` | No | `["https://app.pitchguard.com"]` |
| `RATE_LIMIT_PER_MINUTE` | API rate limit | `100` | No | `200` |

## Model Configuration

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `ARTIFACT_DIR` | Model artifacts directory | `./models` | No | `/app/models` |
| `MODEL_VERSION` | Current model version | `latest` | No | `v1.2.3` |
| `MODEL_RETRAIN_DAYS` | Days between retraining | `7` | No | `14` |
| `MODEL_CALIBRATION_METHOD` | Probability calibration method | `isotonic` | No | `sigmoid` |
| `FEATURE_CACHE_TTL` | Feature cache TTL (seconds) | `300` | No | `600` |

## Data Configuration

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `DATA_SOURCE` | Data source type | `mock` | No | `baseball_savant` |
| `BASEBALL_SAVANT_API_KEY` | Baseball Savant API key | None | No | `abc123def456` |
| `BASEBALL_SAVANT_BASE_URL` | Baseball Savant API URL | `https://baseballsavant.mlb.com` | No | `https://api.baseballsavant.com` |
| `MOCK_DATA_PATH` | Path to mock data files | `./data/mock` | No | `/app/data/mock` |
| `DEFAULT_DATE_RANGE_DAYS` | Default data fetch range | `90` | No | `180` |
| `DATA_BATCH_SIZE` | Batch size for data processing | `1000` | No | `5000` |

## Feature Engineering Configuration

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `MIN_PITCHES_FOR_VELOCITY` | Min pitches for velocity avg | `20` | No | `15` |
| `MIN_PITCHES_FOR_SPIN` | Min pitches for spin avg | `20` | No | `15` |
| `ROLLING_WINDOWS` | Rolling window days | `[3,7,14,30]` | No | `[3,7,14,30,60]` |
| `INJURY_WINDOW_DAYS` | Injury prediction window | `21` | No | `21` |
| `VELOCITY_DROP_THRESHOLD` | Velocity drop alert threshold | `-2.0` | No | `-1.5` |
| `SPIN_DROP_THRESHOLD` | Spin drop alert threshold | `-100` | No | `-75` |

## Logging Configuration

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `LOG_LEVEL` | Logging level | `INFO` | No | `DEBUG` |
| `LOG_FORMAT` | Log format | `json` | No | `text` |
| `LOG_FILE` | Log file path | None | No | `/app/logs/pitchguard.log` |
| `LOG_MAX_SIZE` | Max log file size (MB) | `100` | No | `200` |
| `LOG_BACKUP_COUNT` | Number of backup log files | `5` | No | `10` |

## Monitoring Configuration

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `METRICS_ENABLED` | Enable metrics collection | `true` | No | `true` |
| `METRICS_PORT` | Metrics server port | `9090` | No | `9090` |
| `HEALTH_CHECK_INTERVAL` | Health check interval (seconds) | `30` | No | `60` |
| `ALERT_WEBHOOK_URL` | Alert webhook URL | None | No | `https://hooks.slack.com/...` |

## Security Configuration

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `SECRET_KEY` | Application secret key | None | Yes | `your-secret-key-here` |
| `JWT_SECRET` | JWT signing secret | None | No | `jwt-secret-key` |
| `JWT_ALGORITHM` | JWT algorithm | `HS256` | No | `HS256` |
| `JWT_EXPIRY_HOURS` | JWT token expiry | `24` | No | `48` |

## External Services

### Redis (Caching)

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `REDIS_URL` | Redis connection URL | None | No | `redis://localhost:6379/0` |
| `REDIS_HOST` | Redis host | `localhost` | No | `redis.pitchguard.com` |
| `REDIS_PORT` | Redis port | `6379` | No | `6379` |
| `REDIS_DB` | Redis database number | `0` | No | `1` |
| `REDIS_PASSWORD` | Redis password | None | No | `redis_password` |

### AWS S3 (Model Storage)

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key | None | No | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | None | No | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `AWS_REGION` | AWS region | `us-east-1` | No | `us-west-2` |
| `S3_BUCKET` | S3 bucket for models | None | No | `pitchguard-models` |

## Environment-Specific Configurations

### Development Environment

```bash
# .env.development
NODE_ENV=development
API_RELOAD=true
LOG_LEVEL=DEBUG
DATA_SOURCE=mock
SQLITE_PATH=./data/pitchguard_dev.db
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

### Staging Environment

```bash
# .env.staging
NODE_ENV=staging
API_HOST=0.0.0.0
API_PORT=8000
PG_URL=postgresql://user:pass@staging-db:5432/pitchguard_staging
DATA_SOURCE=baseball_savant
LOG_LEVEL=INFO
METRICS_ENABLED=true
```

### Production Environment

```bash
# .env.production
NODE_ENV=production
API_HOST=0.0.0.0
API_PORT=8000
PG_URL=postgresql://user:pass@prod-db:5432/pitchguard_prod
DATA_SOURCE=baseball_savant
LOG_LEVEL=WARNING
METRICS_ENABLED=true
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your-production-secret-key
```

## Sample .env.example File

```bash
# Database Configuration
PG_URL=postgresql://pitchguard_user:password@localhost:5432/pitchguard
PG_POOL_SIZE=10
PG_MAX_OVERFLOW=20

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=false
CORS_ORIGINS=["http://localhost:3000"]
RATE_LIMIT_PER_MINUTE=100

# Model Configuration
ARTIFACT_DIR=./models
MODEL_VERSION=latest
MODEL_RETRAIN_DAYS=7
MODEL_CALIBRATION_METHOD=isotonic
FEATURE_CACHE_TTL=300

# Data Configuration
DATA_SOURCE=mock
BASEBALL_SAVANT_API_KEY=your_api_key_here
MOCK_DATA_PATH=./data/mock
DEFAULT_DATE_RANGE_DAYS=90
DATA_BATCH_SIZE=1000

# Feature Engineering
MIN_PITCHES_FOR_VELOCITY=20
MIN_PITCHES_FOR_SPIN=20
ROLLING_WINDOWS=[3,7,14,30]
INJURY_WINDOW_DAYS=21
VELOCITY_DROP_THRESHOLD=-2.0
SPIN_DROP_THRESHOLD=-100

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=./logs/pitchguard.log
LOG_MAX_SIZE=100
LOG_BACKUP_COUNT=5

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24

# Redis (Optional)
REDIS_URL=redis://localhost:6379/0

# AWS S3 (Optional)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET=pitchguard-models
```

## Validation Rules

### Required Variables
- `PG_URL` or `SQLITE_PATH` (database connection)
- `SECRET_KEY` (application security)

### Optional but Recommended
- `ARTIFACT_DIR` (model storage)
- `LOG_LEVEL` (logging configuration)
- `DATA_SOURCE` (data source selection)

### Environment-Specific Requirements
- **Development**: `API_RELOAD=true`, `LOG_LEVEL=DEBUG`
- **Production**: `SECRET_KEY`, `PG_URL`, `LOG_LEVEL=WARNING`

## Security Notes

1. **Never commit secrets**: Use `.env` files for local development and secure secret management for production
2. **Rotate secrets regularly**: Update API keys and passwords periodically
3. **Use strong passwords**: Generate secure random strings for secrets
4. **Limit access**: Use least-privilege access for database and external services
5. **Monitor usage**: Track API usage and database connections for anomalies
