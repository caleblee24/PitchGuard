# PitchGuard Backend Design

## Architecture Overview

PitchGuard follows a modular, event-driven architecture optimized for real-time injury risk assessment and workload monitoring.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   ETL Pipeline  │    │   Feature Store │
│                 │    │                 │    │                 │
│ • Baseball      │───▶│ • Ingestion     │───▶│ • Pitches       │
│   Savant API    │    │ • Aggregation   │    │ • Appearances   │
│ • Mock Data     │    │ • Validation    │    │ • Features      │
│ • IL Data       │    │ • Transformation│    │ • Labels        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Layer     │◀───│   Model Layer   │◀───│   Model Registry│
│                 │    │                 │    │                 │
│ • FastAPI       │    │ • Training      │    │ • Versioning    │
│ • Risk Endpoints│    │ • Prediction    │    │ • Metrics       │
│ • Workload API  │    │ • Calibration   │    │ • Artifacts     │
│ • Error Handling│    │ • Explainability│    │ • Performance   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Data Layer

**Purpose**: Store and manage all baseball data and derived features
**Technology**: PostgreSQL with TimescaleDB extension for time-series optimization

#### Tables:
- `pitches`: Raw pitch-level data
- `appearances`: Aggregated game-level data
- `injuries`: IL stint tracking
- `feature_snapshots`: Computed features for ML
- `model_registry`: Model versions and performance

#### Key Design Decisions:
- **Time-series optimization**: Use TimescaleDB for efficient time-range queries
- **Partitioning**: Partition by year for performance
- **Indexing**: Composite indexes on (pitcher_id, game_date) for fast lookups
- **Data retention**: Keep 5 years of historical data

### 2. ETL Pipeline

**Purpose**: Transform raw data into ML-ready features
**Technology**: Python with pandas, SQLAlchemy, and custom transformers

#### Pipeline Stages:

1. **Ingestion** (`etl/ingestion/`)
   - Baseball Savant API client
   - Data validation and cleaning
   - Incremental loading with deduplication

2. **Aggregation** (`etl/aggregation/`)
   - Pitch → Appearance aggregation
   - Velocity/spin rate averaging
   - Outlier detection and handling

3. **Feature Engineering** (`etl/features/`)
   - Rolling window calculations
   - Velocity/spin trend analysis
   - Recovery pattern computation
   - Data completeness assessment

4. **Labeling** (`etl/labeling/`)
   - 21-day forward-looking injury windows
   - Leakage prevention checks
   - Class balance monitoring

#### Key Features:
- **Incremental processing**: Only process new data
- **Data quality checks**: Validation at each stage
- **Error handling**: Graceful failure with retry logic
- **Monitoring**: Comprehensive logging and metrics

### 3. Model Layer

**Purpose**: Train and serve injury risk prediction models
**Technology**: scikit-learn with custom calibration and explainability

#### Model Architecture:

```python
class InjuryRiskModel:
    def __init__(self):
        self.feature_processor = FeatureProcessor()
        self.classifier = LogisticRegression()
        self.calibrator = CalibratedClassifierCV()
        self.explainer = SHAPExplainer()
    
    def train(self, features, labels):
        # Time-based train/val split
        # Feature selection and scaling
        # Model training with cross-validation
        # Probability calibration
        # Performance evaluation
        pass
    
    def predict(self, features):
        # Feature preprocessing
        # Risk score prediction
        # Probability calibration
        # Explainability generation
        pass
```

#### Key Features:
- **Time-aware splitting**: Prevent data leakage
- **Feature selection**: Use domain knowledge and statistical tests
- **Calibration**: Ensure probability scores are well-calibrated
- **Explainability**: SHAP-based feature importance
- **Model versioning**: Track performance and rollback capability

### 4. API Layer

**Purpose**: Serve predictions and workload data to frontend
**Technology**: FastAPI with async/await for high concurrency

#### API Design:

```python
@router.post("/risk/pitcher")
async def get_pitcher_risk(
    request: RiskRequest,
    model_service: ModelService = Depends()
) -> RiskResponse:
    """Get injury risk assessment for a pitcher."""
    pass

@router.get("/workload/pitcher/{pitcher_id}")
async def get_pitcher_workload(
    pitcher_id: int,
    start_date: date,
    end_date: date,
    data_service: DataService = Depends()
) -> WorkloadResponse:
    """Get workload history for a pitcher."""
    pass

@router.get("/pitchers")
async def get_pitchers(
    team: Optional[str] = None,
    season: Optional[int] = None,
    data_service: DataService = Depends()
) -> List[PitcherResponse]:
    """Get list of pitchers with optional filtering."""
    pass
```

#### Key Features:
- **Async processing**: Handle concurrent requests efficiently
- **Caching**: Cache feature computations and model predictions
- **Rate limiting**: Prevent API abuse
- **Error handling**: Comprehensive error codes and messages
- **Validation**: Pydantic models for request/response validation

## Data Flow

### 1. Daily ETL Process

```
1. Fetch new pitch data from Baseball Savant API
2. Validate and clean incoming data
3. Aggregate pitches to appearances
4. Compute rolling features for all pitchers
5. Generate labels for training data
6. Update feature store
7. Trigger model retraining if needed
```

### 2. Risk Assessment Flow

```
1. Receive API request for pitcher risk
2. Fetch latest features for pitcher
3. Validate data completeness
4. Generate model prediction
5. Apply probability calibration
6. Generate explainability insights
7. Return structured response
```

### 3. Model Training Flow

```
1. Extract features and labels from database
2. Perform time-based train/val split
3. Train logistic regression model
4. Calibrate probability outputs
5. Evaluate performance metrics
6. Generate explainability analysis
7. Save model artifacts and metadata
8. Update model registry
```

## Performance Considerations

### Database Optimization
- **Connection pooling**: Use SQLAlchemy connection pools
- **Query optimization**: Use appropriate indexes and query patterns
- **Caching**: Redis for frequently accessed data
- **Partitioning**: Time-based partitioning for large tables

### API Performance
- **Async processing**: Use FastAPI async/await
- **Response caching**: Cache API responses for 5 minutes
- **Batch processing**: Process multiple pitchers efficiently
- **Connection limits**: Limit concurrent database connections

### Model Performance
- **Feature caching**: Cache computed features
- **Model loading**: Keep models in memory
- **Prediction batching**: Batch predictions when possible
- **Monitoring**: Track prediction latency and accuracy

## Security & Reliability

### Data Security
- **Input validation**: Validate all API inputs
- **SQL injection prevention**: Use parameterized queries
- **Rate limiting**: Prevent API abuse
- **Error handling**: Don't expose sensitive information

### System Reliability
- **Health checks**: Monitor system health
- **Circuit breakers**: Handle external API failures
- **Retry logic**: Retry failed operations
- **Logging**: Comprehensive logging for debugging

### Data Quality
- **Validation**: Validate data at each pipeline stage
- **Monitoring**: Track data quality metrics
- **Alerting**: Alert on data quality issues
- **Backup**: Regular database backups

## Monitoring & Observability

### Metrics to Track
- **API performance**: Response times, error rates
- **Model performance**: Prediction accuracy, calibration
- **Data quality**: Completeness, freshness, validity
- **System health**: CPU, memory, disk usage

### Logging Strategy
- **Structured logging**: JSON format for easy parsing
- **Log levels**: DEBUG, INFO, WARNING, ERROR
- **Context**: Include request IDs, user info, timestamps
- **Retention**: Keep logs for 30 days

### Alerting
- **Critical alerts**: System down, data pipeline failures
- **Warning alerts**: High error rates, model performance degradation
- **Info alerts**: Successful deployments, data updates

## Development Workflow

### Local Development
1. **Environment setup**: Use Docker Compose for local development
2. **Mock data**: Use realistic mock data for development
3. **Hot reloading**: FastAPI auto-reload for development
4. **Testing**: Comprehensive unit and integration tests

### Testing Strategy
- **Unit tests**: Test individual components
- **Integration tests**: Test API endpoints
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Test under load

### Deployment
- **Staging environment**: Test changes before production
- **Blue-green deployment**: Zero-downtime deployments
- **Rollback capability**: Quick rollback to previous version
- **Monitoring**: Monitor deployment success

## Future Enhancements

### Phase 2 Features
- **Real-time streaming**: Process data in real-time
- **Advanced models**: XGBoost, neural networks
- **Biomechanics integration**: Add biomechanical data
- **Wearable integration**: Add wearable device data

### Scalability Improvements
- **Microservices**: Split into smaller services
- **Message queues**: Use Kafka for data processing
- **Distributed computing**: Use Spark for large-scale processing
- **Cloud deployment**: Deploy to AWS/GCP

### Advanced Analytics
- **Anomaly detection**: Detect unusual patterns
- **Recommendation engine**: Suggest optimal workloads
- **Predictive maintenance**: Predict equipment failures
- **Performance optimization**: Optimize for specific metrics
