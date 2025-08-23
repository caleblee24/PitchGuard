# Database Schema Overview

## Overview

The PitchGuard database uses PostgreSQL with TimescaleDB extension for efficient time-series data management. The schema is designed to support real-time injury risk assessment with optimized queries for time-range operations.

## Database Architecture

### Core Tables

```
pitches (time-series)
├── appearances (aggregated)
├── injuries (reference)
├── feature_snapshots (time-series)
└── model_registry (metadata)
```

### Technology Stack
- **Database**: PostgreSQL 14+
- **Extension**: TimescaleDB for time-series optimization
- **Connection**: SQLAlchemy with async support
- **Migration**: Alembic for schema versioning

## Table Definitions

### 1. Pitches Table

**Purpose**: Store raw pitch-level data from MLB games
**Type**: Time-series hypertable (TimescaleDB)

```sql
CREATE TABLE pitches (
    id BIGSERIAL PRIMARY KEY,
    game_date DATE NOT NULL,
    pitcher_id INTEGER NOT NULL,
    pitch_type VARCHAR(10) NOT NULL,
    release_speed DECIMAL(4,1),
    release_spin_rate INTEGER,
    pitch_number INTEGER NOT NULL,
    at_bat_id BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('pitches', 'game_date', chunk_time_interval => INTERVAL '1 month');

-- Indexes for performance
CREATE INDEX idx_pitches_pitcher_date ON pitches (pitcher_id, game_date DESC);
CREATE INDEX idx_pitches_at_bat ON pitches (at_bat_id);
CREATE INDEX idx_pitches_type ON pitches (pitch_type);
```

**Constraints**:
- `release_speed`: 50-110 MPH (reasonable MLB range)
- `release_spin_rate`: 0-3500 RPM
- `pitch_number`: > 0 within at_bat
- `game_date`: >= 2015-01-01 (Statcast era)

### 2. Appearances Table

**Purpose**: Aggregated data per pitcher per game appearance
**Type**: Time-series hypertable

```sql
CREATE TABLE appearances (
    id BIGSERIAL PRIMARY KEY,
    game_date DATE NOT NULL,
    pitcher_id INTEGER NOT NULL,
    pitches_thrown INTEGER NOT NULL,
    avg_vel DECIMAL(4,1),
    avg_spin INTEGER,
    outs_recorded INTEGER,
    innings_pitched DECIMAL(3,1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pitcher_id, game_date)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('appearances', 'game_date', chunk_time_interval => INTERVAL '1 month');

-- Indexes for performance
CREATE INDEX idx_appearances_pitcher_date ON appearances (pitcher_id, game_date DESC);
CREATE INDEX idx_appearances_pitches_thrown ON appearances (pitches_thrown);
```

**Constraints**:
- `pitches_thrown`: > 0
- `avg_vel`: 50-110 MPH if not NULL
- `avg_spin`: 0-3500 RPM if not NULL
- `innings_pitched`: >= 0

### 3. Injuries Table

**Purpose**: Track IL (Injured List) stints for pitchers
**Type**: Reference table

```sql
CREATE TABLE injuries (
    id BIGSERIAL PRIMARY KEY,
    pitcher_id INTEGER NOT NULL,
    il_start DATE NOT NULL,
    il_end DATE,
    stint_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_injuries_pitcher ON injuries (pitcher_id);
CREATE INDEX idx_injuries_dates ON injuries (il_start, il_end);
CREATE INDEX idx_injuries_type ON injuries (stint_type);
```

**Constraints**:
- `il_start`: <= il_end (if il_end is not NULL)
- `stint_type`: Valid injury types

### 4. Feature Snapshots Table

**Purpose**: Computed features for model training and prediction
**Type**: Time-series hypertable

```sql
CREATE TABLE feature_snapshots (
    id BIGSERIAL PRIMARY KEY,
    as_of_date DATE NOT NULL,
    pitcher_id INTEGER NOT NULL,
    roll3g_pitch_count INTEGER DEFAULT 0,
    roll3d_pitch_count INTEGER DEFAULT 0,
    roll7d_pitch_count INTEGER DEFAULT 0,
    roll14d_pitch_count INTEGER DEFAULT 0,
    avg_vel_7d DECIMAL(4,1),
    vel_drop_vs_30d DECIMAL(4,1),
    avg_spin_7d INTEGER,
    spin_drop_vs_30d INTEGER,
    rest_days INTEGER DEFAULT 0,
    label_injury_within_21d BOOLEAN NOT NULL,
    data_completeness VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pitcher_id, as_of_date)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('feature_snapshots', 'as_of_date', chunk_time_interval => INTERVAL '1 month');

-- Indexes for performance
CREATE INDEX idx_features_pitcher_date ON feature_snapshots (pitcher_id, as_of_date DESC);
CREATE INDEX idx_features_label ON feature_snapshots (label_injury_within_21d);
CREATE INDEX idx_features_completeness ON feature_snapshots (data_completeness);
```

**Constraints**:
- All rolling counts: >= 0
- `data_completeness`: in ['high', 'med', 'low']
- `rest_days`: >= 0

### 5. Model Registry Table

**Purpose**: Track trained models and their performance
**Type**: Metadata table

```sql
CREATE TABLE model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    framework VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL,
    artifact_ref VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_models_name ON model_registry (model_name);
CREATE INDEX idx_models_active ON model_registry (is_active);
CREATE INDEX idx_models_created ON model_registry (created_at DESC);
```

**Constraints**:
- Only one active model per model_name
- `artifact_ref`: Valid file path or S3 URL

## Relationships

### Foreign Key Constraints

```sql
-- Appearances reference pitchers (pitcher_id)
ALTER TABLE appearances 
ADD CONSTRAINT fk_appearances_pitcher 
FOREIGN KEY (pitcher_id) REFERENCES pitchers(pitcher_id);

-- Injuries reference pitchers (pitcher_id)
ALTER TABLE injuries 
ADD CONSTRAINT fk_injuries_pitcher 
FOREIGN KEY (pitcher_id) REFERENCES pitchers(pitcher_id);

-- Feature snapshots reference pitchers (pitcher_id)
ALTER TABLE feature_snapshots 
ADD CONSTRAINT fk_features_pitcher 
FOREIGN KEY (pitcher_id) REFERENCES pitchers(pitcher_id);
```

### Data Flow Relationships

1. **Pitches → Appearances**: Aggregation by pitcher_id and game_date
2. **Appearances → Feature Snapshots**: Rolling window calculations
3. **Injuries → Feature Snapshots**: Label generation (21-day forward window)
4. **Feature Snapshots → Model Registry**: Training data and performance tracking

## Indexing Strategy

### Primary Indexes

1. **Time-series indexes**: All time-series tables use game_date/as_of_date as primary time dimension
2. **Composite indexes**: (pitcher_id, date) for efficient pitcher-specific queries
3. **Query-specific indexes**: Based on common query patterns

### Query Optimization

```sql
-- Common query: Get pitcher workload for date range
SELECT * FROM appearances 
WHERE pitcher_id = ? AND game_date BETWEEN ? AND ?
ORDER BY game_date DESC;

-- Common query: Get features for model training
SELECT * FROM feature_snapshots 
WHERE as_of_date BETWEEN ? AND ? AND data_completeness = 'high'
ORDER BY as_of_date;

-- Common query: Get injury history for pitcher
SELECT * FROM injuries 
WHERE pitcher_id = ? AND il_start >= ?
ORDER BY il_start DESC;
```

## Partitioning Strategy

### TimescaleDB Chunking

- **Chunk interval**: 1 month for all time-series tables
- **Compression**: Enable compression after 7 days
- **Retention**: Keep 5 years of data

```sql
-- Enable compression
ALTER TABLE pitches SET (timescaledb.compress, timescaledb.compress_segmentby = 'pitcher_id');
ALTER TABLE appearances SET (timescaledb.compress, timescaledb.compress_segmentby = 'pitcher_id');
ALTER TABLE feature_snapshots SET (timescaledb.compress, timescaledb.compress_segmentby = 'pitcher_id');

-- Set compression policy
SELECT add_compression_policy('pitches', INTERVAL '7 days');
SELECT add_compression_policy('appearances', INTERVAL '7 days');
SELECT add_compression_policy('feature_snapshots', INTERVAL '7 days');

-- Set retention policy
SELECT add_retention_policy('pitches', INTERVAL '5 years');
SELECT add_retention_policy('appearances', INTERVAL '5 years');
SELECT add_retention_policy('feature_snapshots', INTERVAL '5 years');
```

## Data Quality Constraints

### Check Constraints

```sql
-- Pitches table constraints
ALTER TABLE pitches ADD CONSTRAINT chk_release_speed 
CHECK (release_speed IS NULL OR (release_speed >= 50 AND release_speed <= 110));

ALTER TABLE pitches ADD CONSTRAINT chk_release_spin 
CHECK (release_spin_rate IS NULL OR (release_spin_rate >= 0 AND release_spin_rate <= 3500));

ALTER TABLE pitches ADD CONSTRAINT chk_pitch_number 
CHECK (pitch_number > 0);

ALTER TABLE pitches ADD CONSTRAINT chk_game_date 
CHECK (game_date >= '2015-01-01');

-- Appearances table constraints
ALTER TABLE appearances ADD CONSTRAINT chk_pitches_thrown 
CHECK (pitches_thrown > 0);

ALTER TABLE appearances ADD CONSTRAINT chk_avg_vel 
CHECK (avg_vel IS NULL OR (avg_vel >= 50 AND avg_vel <= 110));

ALTER TABLE appearances ADD CONSTRAINT chk_avg_spin 
CHECK (avg_spin IS NULL OR (avg_spin >= 0 AND avg_spin <= 3500));

ALTER TABLE appearances ADD CONSTRAINT chk_innings_pitched 
CHECK (innings_pitched IS NULL OR innings_pitched >= 0);

-- Feature snapshots table constraints
ALTER TABLE feature_snapshots ADD CONSTRAINT chk_rolling_counts 
CHECK (roll3g_pitch_count >= 0 AND roll3d_pitch_count >= 0 AND 
       roll7d_pitch_count >= 0 AND roll14d_pitch_count >= 0);

ALTER TABLE feature_snapshots ADD CONSTRAINT chk_data_completeness 
CHECK (data_completeness IN ('high', 'med', 'low'));

ALTER TABLE feature_snapshots ADD CONSTRAINT chk_rest_days 
CHECK (rest_days >= 0);

-- Injuries table constraints
ALTER TABLE injuries ADD CONSTRAINT chk_injury_dates 
CHECK (il_end IS NULL OR il_start <= il_end);
```

## Performance Considerations

### Query Optimization

1. **Use time-range queries**: Leverage TimescaleDB time-series optimization
2. **Limit result sets**: Use LIMIT and OFFSET for pagination
3. **Avoid SELECT ***: Specify only needed columns
4. **Use prepared statements**: For repeated queries with different parameters

### Connection Management

1. **Connection pooling**: Use SQLAlchemy connection pools
2. **Connection limits**: Set appropriate max_connections
3. **Query timeouts**: Set statement_timeout for long-running queries
4. **Monitoring**: Track slow queries and connection usage

### Maintenance

1. **Regular VACUUM**: Clean up dead tuples
2. **ANALYZE**: Update table statistics
3. **REINDEX**: Rebuild indexes periodically
4. **Monitoring**: Track table sizes and growth

## Backup Strategy

### Backup Types

1. **Full backups**: Daily full database backups
2. **Incremental backups**: Hourly incremental backups
3. **WAL archiving**: Continuous WAL archiving for point-in-time recovery

### Recovery Procedures

1. **Point-in-time recovery**: Restore to specific timestamp
2. **Table-level recovery**: Restore individual tables
3. **Data validation**: Verify data integrity after recovery

## Migration Strategy

### Schema Changes

1. **Backward compatibility**: Maintain compatibility during migrations
2. **Rollback plan**: Ability to rollback schema changes
3. **Data migration**: Handle data transformation during schema changes
4. **Testing**: Test migrations in staging environment

### Version Control

1. **Alembic migrations**: Track schema changes in version control
2. **Migration scripts**: Automated migration scripts
3. **Rollback scripts**: Automated rollback scripts
4. **Documentation**: Document all schema changes
