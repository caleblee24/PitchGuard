# API Contracts

## Overview

The PitchGuard API provides RESTful endpoints for injury risk assessment, workload monitoring, and pitcher management. All endpoints return JSON responses and use standard HTTP status codes.

## Base URL

- **Development**: `http://localhost:8000`
- **Staging**: `https://api-staging.pitchguard.com`
- **Production**: `https://api.pitchguard.com`

## Authentication

Currently, the API does not require authentication for MVP. Future versions will implement JWT-based authentication.

## Common Response Format

All successful responses follow this format:

```json
{
  "success": true,
  "data": { ... },
  "timestamp": "2024-04-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

Error responses follow this format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": { ... }
  },
  "timestamp": "2024-04-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

## Endpoints

### 1. Risk Assessment

#### POST /api/v1/risk/pitcher

Get injury risk assessment for a specific pitcher.

**Request Body:**
```json
{
  "pitcher_id": 12345,
  "as_of_date": "2024-04-15"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "pitcher_id": 12345,
    "as_of_date": "2024-04-15",
    "risk_score": 0.23,
    "risk_percentage": 23,
    "risk_level": "medium",
    "confidence": 0.85,
    "data_completeness": "high",
    "top_contributors": [
      {
        "name": "High recent workload",
        "value": 245,
        "direction": "increasing",
        "explanation": "Pitcher threw 245 pitches in last 3 games",
        "mitigation": "Consider +1 rest day before next start"
      },
      {
        "name": "Velocity decline",
        "value": -2.1,
        "direction": "decreasing",
        "explanation": "Average velocity down 2.1 MPH vs 30-day baseline",
        "mitigation": "Monitor velocity in bullpen sessions"
      },
      {
        "name": "Short rest",
        "value": 3,
        "direction": "increasing",
        "explanation": "Only 3 days rest since last appearance",
        "mitigation": "Consider skipping next start"
      }
    ],
    "workload_summary": {
      "last_3_games_pitches": 245,
      "last_7_days_pitches": 180,
      "avg_velocity_7d": 93.2,
      "velocity_drop_vs_30d": -2.1,
      "rest_days": 3
    }
  },
  "timestamp": "2024-04-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

**Error Responses:**

```json
// Pitcher not found
{
  "success": false,
  "error": {
    "code": "UNKNOWN_PITCHER",
    "message": "Pitcher with ID 12345 not found",
    "details": {
      "pitcher_id": 12345
    }
  }
}

// No features available for date
{
  "success": false,
  "error": {
    "code": "NO_FEATURES_FOR_DATE",
    "message": "No feature data available for pitcher 12345 on 2024-04-15",
    "details": {
      "pitcher_id": 12345,
      "as_of_date": "2024-04-15",
      "suggestion": "Try a date within the last 30 days"
    }
  }
}

// Invalid date
{
  "success": false,
  "error": {
    "code": "INVALID_DATE",
    "message": "Date 2024-04-15 is in the future",
    "details": {
      "as_of_date": "2024-04-15",
      "current_date": "2024-04-10"
    }
  }
}
```

### 2. Workload History

#### GET /api/v1/workload/pitcher/{pitcher_id}

Get workload history for a pitcher over a date range.

**Query Parameters:**
- `start_date` (required): Start date in YYYY-MM-DD format
- `end_date` (required): End date in YYYY-MM-DD format
- `include_risk` (optional): Include risk scores in response (default: false)

**Example Request:**
```
GET /api/v1/workload/pitcher/12345?start_date=2024-03-15&end_date=2024-04-15&include_risk=true
```

**Response:**
```json
{
  "success": true,
  "data": {
    "pitcher_id": 12345,
    "date_range": {
      "start": "2024-03-15",
      "end": "2024-04-15"
    },
    "workload_series": [
      {
        "date": "2024-04-15",
        "appearance": {
          "pitches_thrown": 85,
          "avg_vel": 93.2,
          "avg_spin": 2350,
          "innings_pitched": 5.0
        },
        "risk_score": 0.23,
        "risk_level": "medium"
      },
      {
        "date": "2024-04-10",
        "appearance": {
          "pitches_thrown": 95,
          "avg_vel": 94.1,
          "avg_spin": 2400,
          "innings_pitched": 6.0
        },
        "risk_score": 0.15,
        "risk_level": "low"
      }
    ],
    "summary_stats": {
      "total_appearances": 8,
      "total_pitches": 720,
      "avg_pitches_per_appearance": 90.0,
      "avg_velocity": 93.8,
      "avg_spin_rate": 2380
    }
  },
  "timestamp": "2024-04-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

**Error Responses:**

```json
// Invalid date range
{
  "success": false,
  "error": {
    "code": "INVALID_DATE_RANGE",
    "message": "Start date must be before end date",
    "details": {
      "start_date": "2024-04-15",
      "end_date": "2024-04-10"
    }
  }
}

// Date range too large
{
  "success": false,
  "error": {
    "code": "DATE_RANGE_TOO_LARGE",
    "message": "Date range cannot exceed 365 days",
    "details": {
      "start_date": "2023-04-15",
      "end_date": "2024-04-15",
      "days": 366
    }
  }
}
```

### 3. Pitcher List

#### GET /api/v1/pitchers

Get list of pitchers with optional filtering.

**Query Parameters:**
- `team` (optional): Filter by team abbreviation (e.g., "NYY", "LAD")
- `season` (optional): Filter by season year (e.g., 2024)
- `active_only` (optional): Only return active pitchers (default: true)
- `limit` (optional): Maximum number of results (default: 100, max: 1000)
- `offset` (optional): Number of results to skip (default: 0)

**Example Request:**
```
GET /api/v1/pitchers?team=NYY&season=2024&limit=50
```

**Response:**
```json
{
  "success": true,
  "data": {
    "pitchers": [
      {
        "pitcher_id": 12345,
        "name": "Gerrit Cole",
        "team": "NYY",
        "season": 2024,
        "last_appearance": "2024-04-15",
        "current_risk_level": "low",
        "risk_score": 0.12
      },
      {
        "pitcher_id": 12346,
        "name": "Carlos Rodón",
        "team": "NYY",
        "season": 2024,
        "last_appearance": "2024-04-14",
        "current_risk_level": "medium",
        "risk_score": 0.28
      }
    ],
    "pagination": {
      "total": 150,
      "limit": 50,
      "offset": 0,
      "has_more": true
    },
    "filters": {
      "team": "NYY",
      "season": 2024,
      "active_only": true
    }
  },
  "timestamp": "2024-04-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### 4. Team Overview

#### GET /api/v1/teams/{team_id}/overview

Get risk overview for all pitchers on a team.

**Path Parameters:**
- `team_id`: Team abbreviation (e.g., "NYY", "LAD")

**Query Parameters:**
- `season` (optional): Season year (default: current season)
- `include_inactive` (optional): Include inactive pitchers (default: false)

**Example Request:**
```
GET /api/v1/teams/NYY/overview?season=2024
```

**Response:**
```json
{
  "success": true,
  "data": {
    "team_id": "NYY",
    "season": 2024,
    "summary": {
      "total_pitchers": 15,
      "high_risk": 2,
      "medium_risk": 5,
      "low_risk": 8,
      "avg_risk_score": 0.18
    },
    "pitchers": [
      {
        "pitcher_id": 12345,
        "name": "Gerrit Cole",
        "risk_level": "low",
        "risk_score": 0.12,
        "last_appearance": "2024-04-15",
        "trend": "stable",
        "key_concerns": []
      },
      {
        "pitcher_id": 12346,
        "name": "Carlos Rodón",
        "risk_level": "medium",
        "risk_score": 0.28,
        "last_appearance": "2024-04-14",
        "trend": "increasing",
        "key_concerns": [
          "Velocity decline: -1.8 MPH vs 30-day baseline",
          "High recent workload: 280 pitches in last 3 games"
        ]
      }
    ],
    "alerts": [
      {
        "type": "high_risk",
        "message": "2 pitchers showing elevated injury risk",
        "pitchers": [12346, 12347]
      }
    ]
  },
  "timestamp": "2024-04-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### 5. Model Information

#### GET /api/v1/models/current

Get information about the currently active model.

**Response:**
```json
{
  "success": true,
  "data": {
    "model_name": "logistic_regression_v1",
    "version": "1.2.3",
    "created_at": "2024-04-01T10:00:00Z",
    "framework": "scikit-learn",
    "performance_metrics": {
      "auc": 0.78,
      "precision_at_top_10": 0.65,
      "recall_at_top_10": 0.70,
      "brier_score": 0.12,
      "calibration_error": 0.03
    },
    "feature_importance": [
      {
        "feature": "roll7d_pitch_count",
        "importance": 0.25,
        "direction": "positive"
      },
      {
        "feature": "vel_drop_vs_30d",
        "importance": 0.20,
        "direction": "negative"
      }
    ],
    "training_data": {
      "total_samples": 15000,
      "positive_samples": 750,
      "date_range": {
        "start": "2022-01-01",
        "end": "2024-03-31"
      }
    }
  },
  "timestamp": "2024-04-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### 6. Health Check

#### GET /api/v1/health

Check API health and status.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2024-04-15T10:30:00Z",
    "services": {
      "database": "healthy",
      "model_service": "healthy",
      "feature_store": "healthy"
    },
    "uptime": "7 days, 3 hours, 45 minutes",
    "requests_processed": 15420
  },
  "timestamp": "2024-04-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNKNOWN_PITCHER` | 404 | Pitcher not found |
| `NO_FEATURES_FOR_DATE` | 404 | No feature data available for date |
| `INVALID_DATE` | 400 | Invalid date format or future date |
| `INVALID_DATE_RANGE` | 400 | Start date after end date |
| `DATE_RANGE_TOO_LARGE` | 400 | Date range exceeds maximum allowed |
| `INVALID_TEAM` | 400 | Invalid team abbreviation |
| `INVALID_SEASON` | 400 | Invalid season year |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Rate Limiting

- **Default**: 100 requests per minute per IP
- **Burst**: 10 requests per second
- **Headers**: Rate limit information included in response headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1649934000
```

## Request/Response Headers

### Standard Headers

**Request Headers:**
```
Content-Type: application/json
Accept: application/json
User-Agent: PitchGuard-Client/1.0
```

**Response Headers:**
```
Content-Type: application/json
X-Request-ID: req_123456789
X-API-Version: 1.0.0
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
```

## Data Types

### Common Data Types

- **pitcher_id**: Integer (positive)
- **date**: String in YYYY-MM-DD format
- **risk_score**: Float between 0.0 and 1.0
- **risk_percentage**: Integer between 0 and 100
- **risk_level**: String ("low", "medium", "high")
- **velocity**: Float between 50.0 and 110.0 (MPH)
- **spin_rate**: Integer between 0 and 3500 (RPM)
- **pitch_count**: Integer (positive)

### Enums

**Risk Levels:**
- `low`: 0-20% risk
- `medium`: 21-50% risk
- `high`: 51-100% risk

**Data Completeness:**
- `high`: Complete data available
- `med`: Medium completeness
- `low`: Low completeness

**Trends:**
- `increasing`: Risk trending up
- `decreasing`: Risk trending down
- `stable`: Risk stable

## Pagination

For endpoints that return lists, pagination is supported:

**Query Parameters:**
- `limit`: Number of results per page (default: 100, max: 1000)
- `offset`: Number of results to skip (default: 0)

**Response Format:**
```json
{
  "data": [...],
  "pagination": {
    "total": 1500,
    "limit": 100,
    "offset": 0,
    "has_more": true
  }
}
```

## Caching

- **Risk assessments**: Cached for 5 minutes
- **Workload history**: Cached for 1 hour
- **Pitcher lists**: Cached for 30 minutes
- **Model information**: Cached for 24 hours

Cache headers included in responses:
```
Cache-Control: public, max-age=300
ETag: "abc123def456"
```

## Versioning

API versioning is handled through the URL path:
- Current version: `/api/v1/`
- Future versions: `/api/v2/`, `/api/v3/`, etc.

Breaking changes will only occur in new major versions. Minor versions may add new fields but will maintain backward compatibility.
