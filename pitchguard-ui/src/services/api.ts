import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types for our API responses
export interface Pitcher {
  pitcher_id: number;
  name: string;
  team: string;
  role: string;
  last_appearance: string | null;
  current_risk_level: 'low' | 'medium' | 'high';
  current_risk_score: number;
  season_appearances: number;
  recent_velocity: number | null;
  velocity_trend: 'increasing' | 'decreasing' | 'stable';
}

export interface RiskAssessment {
  pitcher_id: number;
  as_of_date: string;
  risk_level: 'low' | 'medium' | 'high';
  risk_score: number;
  confidence: number;
  risk_factors: Array<{
    factor: string;
    value: number;
    importance: number;
    description: string;
  }>;
  primary_concerns: string[];
  recommendations: string[];
  data_completeness: string;
  days_of_data: number;
  last_appearance: string | null;
}

export interface WorkloadDataPoint {
  date: string;
  pitches: number;
  velocity: number;
  spin_rate: number;
  rest_days: number;
  innings: number;
}

export interface WorkloadTimeSeries {
  pitcher_id: number;
  start_date: string;
  end_date: string;
  data_points: WorkloadDataPoint[];
  total_pitches: number;
  avg_pitches_per_game: number;
  avg_velocity: number;
  avg_rest_days: number;
  pitch_count_trend: 'increasing' | 'decreasing' | 'stable';
  velocity_trend: 'increasing' | 'decreasing' | 'stable';
  workload_intensity: string;
}

// API service functions
export const apiService = {
  // Health check
  async getHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  // Get all pitchers
  async getPitchers(params?: {
    team?: string;
    role?: string;
    limit?: number;
    offset?: number;
  }): Promise<Pitcher[]> {
    const response = await api.get('/pitchers', { params });
    // Transform the data to match our interface
    return response.data.map((pitcher: any) => ({
      ...pitcher,
      // Ensure all required fields are present
      last_appearance: pitcher.last_appearance || null,
      current_risk_level: pitcher.current_risk_level || 'medium',
      current_risk_score: pitcher.current_risk_score || 0.3,
      season_appearances: pitcher.season_appearances || 0,
      recent_velocity: pitcher.recent_velocity || null,
      velocity_trend: pitcher.velocity_trend || 'stable',
    }));
  },

  // Get pitcher detail
  async getPitcherDetail(pitcherId: number) {
    const response = await api.get(`/pitchers/${pitcherId}`);
    return response.data;
  },

  // Get current risk assessment for a pitcher
  async getCurrentRisk(pitcherId: number): Promise<RiskAssessment> {
    const response = await api.get(`/risk/pitcher/${pitcherId}/current`);
    return response.data;
  },

  // Get risk assessment for a specific date
  async getRiskAssessment(pitcherId: number, asOfDate: string): Promise<RiskAssessment> {
    const response = await api.post('/risk/pitcher', {
      pitcher_id: pitcherId,
      as_of_date: asOfDate,
    });
    return response.data;
  },

  // Get workload time series
  async getWorkloadTimeSeries(
    pitcherId: number,
    startDate?: string,
    endDate?: string
  ): Promise<WorkloadTimeSeries> {
    const params: any = {};
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    
    const response = await api.get(`/workload/pitcher/${pitcherId}`, { params });
    return response.data;
  },

  // Get team overview
  async getTeamOverview(teamId: string) {
    const response = await api.get(`/teams/${teamId}/overview`);
    return response.data;
  },

  // Get team risk summary
  async getTeamRiskSummary(teamId: string, asOfDate?: string) {
    const params: any = {};
    if (asOfDate) params.as_of_date = asOfDate;
    
    const response = await api.get(`/risk/team/${teamId}/summary`, { params });
    return response.data;
  },
};

export default apiService;
