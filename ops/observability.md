# Observability Strategy

## Overview

This document outlines the comprehensive observability strategy for PitchGuard, including monitoring, logging, alerting, and performance tracking to ensure system reliability and operational excellence.

## Monitoring Architecture

### Monitoring Stack

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Infrastructure│    │   Business      │
│   Metrics       │    │   Metrics       │    │   Metrics       │
│                 │    │                 │    │                 │
│ • API Response  │    │ • CPU/Memory    │    │ • Risk Scores   │
│ • Model Latency │    │ • Disk Usage    │    │ • Predictions   │
│ • Error Rates   │    │ • Network I/O   │    │ • Data Quality  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Monitoring    │
                    │   Platform      │
                    │                 │
                    │ • Prometheus    │
                    │ • Grafana       │
                    │ • AlertManager  │
                    └─────────────────┘
```

### Key Metrics Categories

1. **Application Metrics**: API performance, model accuracy, data processing
2. **Infrastructure Metrics**: System resources, database performance
3. **Business Metrics**: Risk assessment quality, user engagement
4. **Operational Metrics**: Pipeline health, data freshness

## Application Metrics

### API Performance Metrics

```python
# FastAPI middleware for metrics collection
from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('api_active_requests', 'Number of active requests')

# Model metrics
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total model predictions', ['model_version'])
MODEL_LATENCY = Histogram('model_prediction_duration_seconds', 'Model prediction latency')
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy score')

# Data metrics
DATA_FRESHNESS = Gauge('data_freshness_hours', 'Hours since last data update')
FEATURE_COMPLETENESS = Gauge('feature_completeness_rate', 'Feature data completeness rate')
```

### Model Performance Metrics

```python
def track_model_metrics(model_name, prediction_time, accuracy_score):
    """Track model performance metrics."""
    
    # Record prediction
    MODEL_PREDICTIONS.labels(model_version=model_name).inc()
    
    # Record latency
    MODEL_LATENCY.observe(prediction_time)
    
    # Update accuracy
    MODEL_ACCURACY.set(accuracy_score)

def track_data_quality_metrics():
    """Track data quality metrics."""
    
    # Check data freshness
    last_update = get_last_data_update()
    hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
    DATA_FRESHNESS.set(hours_since_update)
    
    # Check feature completeness
    completeness_rate = calculate_feature_completeness()
    FEATURE_COMPLETENESS.set(completeness_rate)
```

### ETL Pipeline Metrics

```python
# ETL pipeline metrics
ETL_RUNS = Counter('etl_runs_total', 'Total ETL pipeline runs', ['status'])
ETL_DURATION = Histogram('etl_duration_seconds', 'ETL pipeline duration')
ETL_RECORDS_PROCESSED = Counter('etl_records_processed_total', 'Records processed by ETL', ['stage'])

def track_etl_metrics(stage, records_processed, duration, status):
    """Track ETL pipeline metrics."""
    
    # Record ETL run
    ETL_RUNS.labels(status=status).inc()
    
    # Record duration
    ETL_DURATION.observe(duration)
    
    # Record records processed
    ETL_RECORDS_PROCESSED.labels(stage=stage).inc(records_processed)
```

## Infrastructure Metrics

### System Resource Monitoring

```python
# System metrics
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Memory usage in bytes')
DISK_USAGE = Gauge('system_disk_usage_percent', 'Disk usage percentage')
NETWORK_IO = Counter('system_network_bytes_total', 'Network I/O in bytes', ['direction'])

def collect_system_metrics():
    """Collect system resource metrics."""
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    CPU_USAGE.set(cpu_percent)
    
    # Memory usage
    memory = psutil.virtual_memory()
    MEMORY_USAGE.set(memory.used)
    
    # Disk usage
    disk = psutil.disk_usage('/')
    DISK_USAGE.set((disk.used / disk.total) * 100)
    
    # Network I/O
    network = psutil.net_io_counters()
    NETWORK_IO.labels(direction='sent').inc(network.bytes_sent)
    NETWORK_IO.labels(direction='received').inc(network.bytes_recv)
```

### Database Performance Metrics

```python
# Database metrics
DB_CONNECTIONS = Gauge('database_connections_active', 'Active database connections')
DB_QUERY_DURATION = Histogram('database_query_duration_seconds', 'Database query duration', ['query_type'])
DB_ERRORS = Counter('database_errors_total', 'Database errors', ['error_type'])

def track_database_metrics():
    """Track database performance metrics."""
    
    # Active connections
    active_connections = get_active_db_connections()
    DB_CONNECTIONS.set(active_connections)
    
    # Query performance
    query_stats = get_query_performance_stats()
    for query_type, duration in query_stats.items():
        DB_QUERY_DURATION.labels(query_type=query_type).observe(duration)
```

## Business Metrics

### Risk Assessment Quality

```python
# Business metrics
RISK_ASSESSMENTS = Counter('risk_assessments_total', 'Total risk assessments', ['risk_level'])
PREDICTION_ACCURACY = Gauge('prediction_accuracy', 'Prediction accuracy over time')
FALSE_POSITIVES = Counter('false_positives_total', 'False positive predictions')
FALSE_NEGATIVES = Counter('false_negatives_total', 'False negative predictions')

def track_business_metrics(risk_level, prediction_accuracy, false_positives, false_negatives):
    """Track business-relevant metrics."""
    
    # Risk assessments by level
    RISK_ASSESSMENTS.labels(risk_level=risk_level).inc()
    
    # Prediction accuracy
    PREDICTION_ACCURACY.set(prediction_accuracy)
    
    # False predictions
    FALSE_POSITIVES.inc(false_positives)
    FALSE_NEGATIVES.inc(false_negatives)
```

### User Engagement Metrics

```python
# User engagement metrics
ACTIVE_USERS = Gauge('active_users_total', 'Number of active users')
API_USAGE = Counter('api_usage_total', 'API usage by endpoint', ['endpoint'])
USER_SESSIONS = Counter('user_sessions_total', 'User sessions', ['duration_bucket'])

def track_user_metrics(user_id, endpoint, session_duration):
    """Track user engagement metrics."""
    
    # API usage
    API_USAGE.labels(endpoint=endpoint).inc()
    
    # Session duration
    if session_duration < 300:  # 5 minutes
        duration_bucket = 'short'
    elif session_duration < 1800:  # 30 minutes
        duration_bucket = 'medium'
    else:
        duration_bucket = 'long'
    
    USER_SESSIONS.labels(duration_bucket=duration_bucket).inc()
```

## Logging Strategy

### Log Levels and Structure

```python
import logging
import json
from datetime import datetime

# Configure structured logging
def setup_logging():
    """Setup structured logging configuration."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/pitchguard.log'),
            logging.StreamHandler()
        ]
    )

class StructuredLogger:
    """Structured logger for consistent log format."""
    
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_event(self, event_type, message, **kwargs):
        """Log structured event."""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'message': message,
            'level': 'INFO',
            **kwargs
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, error_type, message, error_details=None, **kwargs):
        """Log structured error."""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'ERROR',
            'error_type': error_type,
            'message': message,
            'level': 'ERROR',
            'error_details': error_details,
            **kwargs
        }
        
        self.logger.error(json.dumps(log_entry))
```

### Application Logging

```python
# API logging
def log_api_request(request_id, method, endpoint, status_code, duration, user_id=None):
    """Log API request details."""
    
    logger = StructuredLogger('api')
    logger.log_event(
        'api_request',
        f"API request completed: {method} {endpoint}",
        request_id=request_id,
        method=method,
        endpoint=endpoint,
        status_code=status_code,
        duration_ms=duration * 1000,
        user_id=user_id
    )

# Model logging
def log_model_prediction(model_name, pitcher_id, risk_score, prediction_time, confidence):
    """Log model prediction details."""
    
    logger = StructuredLogger('model')
    logger.log_event(
        'model_prediction',
        f"Risk prediction for pitcher {pitcher_id}",
        model_name=model_name,
        pitcher_id=pitcher_id,
        risk_score=risk_score,
        prediction_time_ms=prediction_time * 1000,
        confidence=confidence
    )

# ETL logging
def log_etl_stage(stage_name, records_processed, duration, status, error_details=None):
    """Log ETL pipeline stage details."""
    
    logger = StructuredLogger('etl')
    if status == 'success':
        logger.log_event(
            'etl_stage_completed',
            f"ETL stage {stage_name} completed successfully",
            stage=stage_name,
            records_processed=records_processed,
            duration_seconds=duration,
            status=status
        )
    else:
        logger.log_error(
            'etl_stage_failed',
            f"ETL stage {stage_name} failed",
            stage=stage_name,
            records_processed=records_processed,
            duration_seconds=duration,
            error_details=error_details
        )
```

### Error Logging

```python
def log_system_error(error_type, error_message, stack_trace=None, context=None):
    """Log system errors with full context."""
    
    logger = StructuredLogger('system')
    logger.log_error(
        error_type,
        error_message,
        error_details={
            'stack_trace': stack_trace,
            'context': context
        }
    )

def log_data_quality_issue(issue_type, description, affected_records, severity):
    """Log data quality issues."""
    
    logger = StructuredLogger('data_quality')
    logger.log_event(
        'data_quality_issue',
        f"Data quality issue detected: {description}",
        issue_type=issue_type,
        description=description,
        affected_records=affected_records,
        severity=severity
    )
```

## Alerting Strategy

### Alert Severity Levels

1. **Critical**: System down, data corruption, security breach
2. **High**: Performance degradation, high error rates
3. **Medium**: Warning thresholds exceeded, data quality issues
4. **Low**: Informational alerts, system maintenance

### Alert Rules

```yaml
# Prometheus alert rules
groups:
  - name: pitchguard_alerts
    rules:
      # Critical alerts
      - alert: APIDown
        expr: up{job="pitchguard-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PitchGuard API is down"
          description: "API has been down for more than 1 minute"

      - alert: DatabaseDown
        expr: up{job="pitchguard-db"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"
          description: "Database connection lost"

      # High severity alerts
      - alert: HighErrorRate
        expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "High API error rate"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "High API response time"
          description: "95th percentile response time is {{ $value }} seconds"

      # Medium severity alerts
      - alert: DataStale
        expr: data_freshness_hours > 24
        for: 1h
        labels:
          severity: medium
        annotations:
          summary: "Data is stale"
          description: "Data hasn't been updated for {{ $value }} hours"

      - alert: LowDataCompleteness
        expr: feature_completeness_rate < 0.8
        for: 1h
        labels:
          severity: medium
        annotations:
          summary: "Low data completeness"
          description: "Feature completeness is {{ $value }}%"

      # Low severity alerts
      - alert: ModelAccuracyDeclining
        expr: model_accuracy < 0.7
        for: 1h
        labels:
          severity: low
        annotations:
          summary: "Model accuracy declining"
          description: "Model accuracy is {{ $value }}%"
```

### Alert Channels

```python
# Alert notification channels
ALERT_CHANNELS = {
    'critical': ['pagerduty', 'slack-critical', 'email-admin'],
    'high': ['slack-alerts', 'email-team'],
    'medium': ['slack-notifications'],
    'low': ['slack-info']
}

def send_alert(alert_name, severity, message, details=None):
    """Send alert to appropriate channels."""
    
    channels = ALERT_CHANNELS.get(severity, ['slack-notifications'])
    
    alert_payload = {
        'alert_name': alert_name,
        'severity': severity,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'details': details
    }
    
    for channel in channels:
        send_to_channel(channel, alert_payload)
```

## Performance Monitoring

### Key Performance Indicators (KPIs)

```python
# Performance KPIs
PERFORMANCE_KPIS = {
    'api_response_time_p95': '95th percentile API response time < 2 seconds',
    'api_availability': 'API availability > 99.9%',
    'model_prediction_latency': 'Model prediction latency < 500ms',
    'data_freshness': 'Data freshness < 24 hours',
    'feature_completeness': 'Feature completeness > 80%',
    'model_accuracy': 'Model accuracy > 70%'
}

def calculate_kpis():
    """Calculate current KPI values."""
    
    kpis = {}
    
    # API response time P95
    response_time_p95 = get_response_time_percentile(0.95)
    kpis['api_response_time_p95'] = response_time_p95
    
    # API availability
    availability = calculate_api_availability()
    kpis['api_availability'] = availability
    
    # Model prediction latency
    prediction_latency = get_model_prediction_latency()
    kpis['model_prediction_latency'] = prediction_latency
    
    # Data freshness
    data_freshness = get_data_freshness_hours()
    kpis['data_freshness'] = data_freshness
    
    # Feature completeness
    feature_completeness = get_feature_completeness_rate()
    kpis['feature_completeness'] = feature_completeness
    
    # Model accuracy
    model_accuracy = get_model_accuracy()
    kpis['model_accuracy'] = model_accuracy
    
    return kpis

def check_kpi_compliance():
    """Check KPI compliance and generate alerts."""
    
    kpis = calculate_kpis()
    violations = []
    
    for kpi_name, current_value in kpis.items():
        threshold = get_kpi_threshold(kpi_name)
        if not is_kpi_compliant(kpi_name, current_value, threshold):
            violations.append({
                'kpi': kpi_name,
                'current_value': current_value,
                'threshold': threshold
            })
    
    return violations
```

### Performance Dashboards

```python
# Grafana dashboard configuration
DASHBOARD_CONFIG = {
    'api_performance': {
        'title': 'API Performance',
        'panels': [
            {
                'title': 'Response Time',
                'type': 'graph',
                'query': 'histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))'
            },
            {
                'title': 'Request Rate',
                'type': 'graph',
                'query': 'rate(api_requests_total[5m])'
            },
            {
                'title': 'Error Rate',
                'type': 'graph',
                'query': 'rate(api_requests_total{status=~"5.."}[5m])'
            }
        ]
    },
    'model_performance': {
        'title': 'Model Performance',
        'panels': [
            {
                'title': 'Prediction Latency',
                'type': 'graph',
                'query': 'histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m]))'
            },
            {
                'title': 'Model Accuracy',
                'type': 'singlestat',
                'query': 'model_accuracy'
            },
            {
                'title': 'Predictions per Hour',
                'type': 'graph',
                'query': 'rate(model_predictions_total[1h])'
            }
        ]
    },
    'data_quality': {
        'title': 'Data Quality',
        'panels': [
            {
                'title': 'Data Freshness',
                'type': 'singlestat',
                'query': 'data_freshness_hours'
            },
            {
                'title': 'Feature Completeness',
                'type': 'singlestat',
                'query': 'feature_completeness_rate'
            },
            {
                'title': 'ETL Pipeline Status',
                'type': 'stat',
                'query': 'etl_runs_total'
            }
        ]
    }
}
```

## Health Checks

### System Health Endpoints

```python
# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {}
    }
    
    # Check database health
    try:
        db_health = check_database_health()
        health_status['services']['database'] = db_health
    except Exception as e:
        health_status['services']['database'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Check model service health
    try:
        model_health = check_model_service_health()
        health_status['services']['model_service'] = model_health
    except Exception as e:
        health_status['services']['model_service'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Check data freshness
    try:
        data_health = check_data_health()
        health_status['services']['data_freshness'] = data_health
    except Exception as e:
        health_status['services']['data_freshness'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Determine overall status
    all_healthy = all(
        service.get('status') == 'healthy' 
        for service in health_status['services'].values()
    )
    
    health_status['status'] = 'healthy' if all_healthy else 'unhealthy'
    
    return health_status

@app.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    
    # Check if system is ready to serve traffic
    ready = check_system_readiness()
    
    if ready:
        return {'status': 'ready'}
    else:
        raise HTTPException(status_code=503, detail="System not ready")

@app.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    
    # Check if system is alive
    alive = check_system_alive()
    
    if alive:
        return {'status': 'alive'}
    else:
        raise HTTPException(status_code=503, detail="System not alive")
```

## Incident Response

### Incident Severity Levels

```python
INCIDENT_SEVERITY = {
    'sev1': {
        'description': 'Critical system failure',
        'response_time': '15 minutes',
        'resolution_time': '1 hour',
        'notification': ['pagerduty', 'slack-critical', 'email-admin']
    },
    'sev2': {
        'description': 'Major functionality impaired',
        'response_time': '30 minutes',
        'resolution_time': '4 hours',
        'notification': ['slack-alerts', 'email-team']
    },
    'sev3': {
        'description': 'Minor functionality impaired',
        'response_time': '2 hours',
        'resolution_time': '24 hours',
        'notification': ['slack-notifications']
    },
    'sev4': {
        'description': 'Informational or enhancement request',
        'response_time': '24 hours',
        'resolution_time': '1 week',
        'notification': ['slack-info']
    }
}
```

### Incident Response Process

```python
def handle_incident(incident_type, severity, description, details=None):
    """Handle incident response."""
    
    incident_id = generate_incident_id()
    
    # Create incident record
    incident = {
        'id': incident_id,
        'type': incident_type,
        'severity': severity,
        'description': description,
        'details': details,
        'created_at': datetime.now().isoformat(),
        'status': 'open'
    }
    
    # Send notifications
    notification_channels = INCIDENT_SEVERITY[severity]['notification']
    for channel in notification_channels:
        send_incident_notification(channel, incident)
    
    # Create incident ticket
    create_incident_ticket(incident)
    
    # Start incident response
    start_incident_response(incident)
    
    return incident_id

def resolve_incident(incident_id, resolution_details):
    """Resolve incident."""
    
    incident = get_incident(incident_id)
    incident['status'] = 'resolved'
    incident['resolved_at'] = datetime.now().isoformat()
    incident['resolution_details'] = resolution_details
    
    # Update incident record
    update_incident(incident)
    
    # Send resolution notification
    send_resolution_notification(incident)
    
    # Close incident ticket
    close_incident_ticket(incident_id)
```

## Data Retention and Archival

### Log Retention Policy

```python
LOG_RETENTION_POLICY = {
    'application_logs': {
        'retention_days': 30,
        'compression_after_days': 7,
        'archival_after_days': 90
    },
    'error_logs': {
        'retention_days': 90,
        'compression_after_days': 30,
        'archival_after_days': 365
    },
    'audit_logs': {
        'retention_days': 365,
        'compression_after_days': 90,
        'archival_after_days': 2555  # 7 years
    }
}

def cleanup_old_logs():
    """Clean up old logs according to retention policy."""
    
    for log_type, policy in LOG_RETENTION_POLICY.items():
        cutoff_date = datetime.now() - timedelta(days=policy['retention_days'])
        
        # Delete old log files
        delete_logs_older_than(log_type, cutoff_date)
        
        # Compress logs
        compression_date = datetime.now() - timedelta(days=policy['compression_after_days'])
        compress_logs_older_than(log_type, compression_date)
        
        # Archive logs
        archival_date = datetime.now() - timedelta(days=policy['archival_after_days'])
        archive_logs_older_than(log_type, archival_date)
```

### Metrics Retention

```python
METRICS_RETENTION_POLICY = {
    'high_resolution': {
        'retention_days': 7,
        'resolution': '1m'
    },
    'medium_resolution': {
        'retention_days': 30,
        'resolution': '5m'
    },
    'low_resolution': {
        'retention_days': 365,
        'resolution': '1h'
    }
}

def configure_metrics_retention():
    """Configure metrics retention in Prometheus."""
    
    retention_config = {
        'storage.tsdb.retention.time': '365d',
        'storage.tsdb.retention.size': '50GB',
        'storage.tsdb.wal-compression': 'true'
    }
    
    return retention_config
```
