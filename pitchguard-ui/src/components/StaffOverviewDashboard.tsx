import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Chip,
  IconButton,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Tooltip,
} from '@mui/material';

import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Visibility,
  Warning,
  CheckCircle,
  Error,
} from '@mui/icons-material';
import { format } from 'date-fns';
import { apiService, Pitcher } from '../services/api';

// Risk level color mapping
const getRiskColor = (level: string) => {
  switch (level) {
    case 'high':
      return 'error';
    case 'medium':
      return 'warning';
    case 'low':
      return 'success';
    default:
      return 'default';
  }
};

// Trend icon mapping
const getTrendIcon = (trend: string) => {
  switch (trend) {
    case 'increasing':
      return <TrendingUp color="error" />;
    case 'decreasing':
      return <TrendingDown color="success" />;
    case 'stable':
      return <TrendingFlat color="info" />;
    default:
      return <TrendingFlat color="disabled" />;
  }
};

// Risk level summary component
const RiskSummary: React.FC<{ pitchers: Pitcher[] }> = ({ pitchers }) => {
  const riskCounts = pitchers.reduce(
    (acc, pitcher) => {
      acc[pitcher.current_risk_level]++;
      return acc;
    },
    { low: 0, medium: 0, high: 0 }
  );

  return (
    <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
      <Box sx={{ flex: '1 1 300px', minWidth: 0 }}>
        <Card sx={{ bgcolor: 'success.light', color: 'white' }}>
          <CardContent>
            <Typography variant="h4" component="div">
              {riskCounts.low}
            </Typography>
            <Typography variant="body2">Low Risk</Typography>
          </CardContent>
        </Card>
      </Box>
      <Box sx={{ flex: '1 1 300px', minWidth: 0 }}>
        <Card sx={{ bgcolor: 'warning.light', color: 'white' }}>
          <CardContent>
            <Typography variant="h4" component="div">
              {riskCounts.medium}
            </Typography>
            <Typography variant="body2">Medium Risk</Typography>
          </CardContent>
        </Card>
      </Box>
      <Box sx={{ flex: '1 1 300px', minWidth: 0 }}>
        <Card sx={{ bgcolor: 'error.light', color: 'white' }}>
          <CardContent>
            <Typography variant="h4" component="div">
              {riskCounts.high}
            </Typography>
            <Typography variant="body2">High Risk</Typography>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

const StaffOverviewDashboard: React.FC = () => {
  const [pitchers, setPitchers] = useState<Pitcher[]>([]);
  const [filteredPitchers, setFilteredPitchers] = useState<Pitcher[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [teamFilter, setTeamFilter] = useState('');
  const [roleFilter, setRoleFilter] = useState('');

  // Load pitchers data
  useEffect(() => {
    const loadPitchers = async () => {
      try {
        setLoading(true);
        const data = await apiService.getPitchers();
        setPitchers(data);
        setFilteredPitchers(data);
      } catch (err) {
        setError('Failed to load pitchers data. Please check your connection.');
        console.error('Error loading pitchers:', err);
      } finally {
        setLoading(false);
      }
    };

    loadPitchers();
  }, []);

  // Apply filters
  useEffect(() => {
    let filtered = pitchers;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(
        (pitcher) =>
          pitcher.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          pitcher.team.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply team filter
    if (teamFilter) {
      filtered = filtered.filter((pitcher) => pitcher.team === teamFilter);
    }

    // Apply role filter
    if (roleFilter) {
      filtered = filtered.filter((pitcher) => pitcher.role === roleFilter);
    }

    setFilteredPitchers(filtered);
  }, [pitchers, searchTerm, teamFilter, roleFilter]);

  // Get unique teams and roles for filters
  const teams = Array.from(new Set(pitchers.map((p) => p.team))).sort();
  const roles = Array.from(new Set(pitchers.map((p) => p.role))).sort();

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Typography variant="h4" component="h1" gutterBottom>
        Staff Overview Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Monitor pitcher workload and injury risk across your organization
      </Typography>

      {/* Risk Summary Cards */}
      <RiskSummary pitchers={filteredPitchers} />

      {/* Filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          <Box sx={{ flex: '1 1 300px', minWidth: 0 }}>
            <TextField
              fullWidth
              label="Search pitchers"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search by name or team..."
            />
          </Box>
          <Box sx={{ flex: '1 1 200px', minWidth: 0 }}>
            <FormControl fullWidth>
              <InputLabel>Team</InputLabel>
              <Select
                value={teamFilter}
                label="Team"
                onChange={(e) => setTeamFilter(e.target.value)}
              >
                <MenuItem value="">All Teams</MenuItem>
                {teams.map((team) => (
                  <MenuItem key={team} value={team}>
                    {team}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
          <Box sx={{ flex: '1 1 200px', minWidth: 0 }}>
            <FormControl fullWidth>
              <InputLabel>Role</InputLabel>
              <Select
                value={roleFilter}
                label="Role"
                onChange={(e) => setRoleFilter(e.target.value)}
              >
                <MenuItem value="">All Roles</MenuItem>
                {roles.map((role) => (
                  <MenuItem key={role} value={role}>
                    {role.charAt(0).toUpperCase() + role.slice(1)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
          <Box sx={{ flex: '0 0 auto' }}>
            <Typography variant="body2" color="text.secondary">
              {filteredPitchers.length} of {pitchers.length} pitchers
            </Typography>
          </Box>
        </Box>
      </Paper>

      {/* Pitchers Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow sx={{ bgcolor: 'grey.100' }}>
              <TableCell>Pitcher</TableCell>
              <TableCell>Team</TableCell>
              <TableCell>Role</TableCell>
              <TableCell>Risk Level</TableCell>
              <TableCell>Risk Score</TableCell>
              <TableCell>Last Appearance</TableCell>
              <TableCell>Season Appearances</TableCell>
              <TableCell>Recent Velocity</TableCell>
              <TableCell>Velocity Trend</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredPitchers.map((pitcher) => (
              <TableRow key={pitcher.pitcher_id} hover>
                <TableCell>
                  <Typography variant="subtitle2" fontWeight="bold">
                    {pitcher.name}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip label={pitcher.team} size="small" />
                </TableCell>
                <TableCell>
                  <Chip
                    label={pitcher.role}
                    size="small"
                    variant="outlined"
                    color={pitcher.role === 'starter' ? 'primary' : 'secondary'}
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    label={pitcher.current_risk_level.toUpperCase()}
                    color={getRiskColor(pitcher.current_risk_level)}
                    size="small"
                    icon={
                      pitcher.current_risk_level === 'high' ? (
                        <Warning />
                      ) : pitcher.current_risk_level === 'medium' ? (
                        <Error />
                      ) : (
                        <CheckCircle />
                      )
                    }
                  />
                </TableCell>
                <TableCell>
                  <Typography
                    variant="body2"
                    color={getRiskColor(pitcher.current_risk_level)}
                    fontWeight="bold"
                  >
                    {(pitcher.current_risk_score * 100).toFixed(0)}%
                  </Typography>
                </TableCell>
                <TableCell>
                  {pitcher.last_appearance ? (
                    <Typography variant="body2">
                      {format(new Date(pitcher.last_appearance), 'MMM d, yyyy')}
                    </Typography>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No data
                    </Typography>
                  )}
                </TableCell>
                <TableCell>
                  <Typography variant="body2">{pitcher.season_appearances}</Typography>
                </TableCell>
                <TableCell>
                  {pitcher.recent_velocity ? (
                    <Typography variant="body2">
                      {pitcher.recent_velocity.toFixed(1)} MPH
                    </Typography>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No data
                    </Typography>
                  )}
                </TableCell>
                <TableCell>
                  <Tooltip title={`Velocity trend: ${pitcher.velocity_trend}`}>
                    <Box>{getTrendIcon(pitcher.velocity_trend)}</Box>
                  </Tooltip>
                </TableCell>
                <TableCell>
                  <Tooltip title="View detailed analysis">
                    <IconButton size="small" color="primary">
                      <Visibility />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Empty state */}
      {filteredPitchers.length === 0 && (
        <Box textAlign="center" py={4}>
          <Typography variant="h6" color="text.secondary">
            No pitchers found matching your filters
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try adjusting your search criteria
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default StaffOverviewDashboard;
