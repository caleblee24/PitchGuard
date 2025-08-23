# Frontend UX Specification

## Overview

The PitchGuard frontend provides an intuitive, data-driven interface for MLB coaching staff to monitor pitcher workload and injury risk. The design prioritizes clarity, actionable insights, and efficient workflow for daily decision-making.

## Design Principles

1. **Clarity First**: Present complex data in easily digestible formats
2. **Actionable Insights**: Every data point should lead to a decision
3. **Progressive Disclosure**: Show summary first, details on demand
4. **Mobile Responsive**: Works on tablets and mobile devices
5. **Accessibility**: WCAG 2.1 AA compliance

## Color Palette

### Primary Colors
- **Primary Blue**: `#1E40AF` (Risk assessment, primary actions)
- **Success Green**: `#059669` (Low risk, positive trends)
- **Warning Yellow**: `#D97706` (Medium risk, caution)
- **Danger Red**: `#DC2626` (High risk, alerts)

### Neutral Colors
- **Background**: `#F8FAFC`
- **Surface**: `#FFFFFF`
- **Border**: `#E2E8F0`
- **Text Primary**: `#1E293B`
- **Text Secondary**: `#64748B`

### Risk Level Colors
- **Low Risk**: `#10B981` (Green)
- **Medium Risk**: `#F59E0B` (Yellow)
- **High Risk**: `#EF4444` (Red)

## Typography

### Font Stack
- **Primary**: Inter (Google Fonts)
- **Fallback**: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif

### Font Sizes
- **Heading 1**: 2.25rem (36px) - Page titles
- **Heading 2**: 1.875rem (30px) - Section headers
- **Heading 3**: 1.5rem (24px) - Subsection headers
- **Body Large**: 1.125rem (18px) - Important text
- **Body**: 1rem (16px) - Regular text
- **Body Small**: 0.875rem (14px) - Secondary text
- **Caption**: 0.75rem (12px) - Labels, timestamps

## Layout Structure

### Main Layout
```
┌─────────────────────────────────────────────────────────┐
│ Header (Logo, Navigation, User Menu)                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Main Content Area                                       │
│ ┌─────────────────┐ ┌─────────────────┐                │
│ │ Sidebar         │ │ Content         │                │
│ │ (Filters,       │ │ (Tables,        │                │
│ │  Quick Stats)   │ │  Charts,        │                │
│ │                 │ │  Details)       │                │
│ └─────────────────┘ └─────────────────┘                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Responsive Breakpoints
- **Desktop**: 1024px+
- **Tablet**: 768px - 1023px
- **Mobile**: < 768px

## Page Specifications

### 1. Staff Overview Dashboard

**Purpose**: Daily overview of all pitchers with risk indicators and quick actions

#### Layout
```
┌─────────────────────────────────────────────────────────┐
│ Page Header: "Pitcher Overview" + Date Selector         │
├─────────────────────────────────────────────────────────┤
│ Filters: Team | Season | Risk Level | Sort Options      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ ┌─────────────────┐ ┌─────────────────┐                │
│ │ Summary Cards   │ │ Risk Alerts     │                │
│ │ (3 cards)       │ │ (High-risk      │                │
│ │                 │ │  pitchers)      │                │
│ └─────────────────┘ └─────────────────┘                │
│                                                         │
│ Pitchers Table (Main Content)                           │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Name | Team | Last App | Risk | Trend | Actions    │ │
│ │ -----|------|----------|------|-------|----------   │ │
│ │ Cole | NYY  | 4/15     | Low  | ↗️    | View Details│ │
│ │ Rodón| NYY  | 4/14     | Med  | ↗️    | View Details│ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Components

**Summary Cards**
```typescript
interface SummaryCard {
  title: string;
  value: number;
  change: number;
  trend: 'up' | 'down' | 'stable';
  color: 'green' | 'yellow' | 'red' | 'blue';
}
```

**Pitchers Table**
```typescript
interface PitcherRow {
  id: number;
  name: string;
  team: string;
  lastAppearance: string;
  riskLevel: 'low' | 'medium' | 'high';
  riskScore: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  keyConcerns: string[];
  actions: Action[];
}
```

#### Interactions
- **Click pitcher row**: Navigate to pitcher detail page
- **Sort by column**: Click column header to sort
- **Filter by risk**: Quick filter buttons
- **Search**: Type to filter by name or team
- **Export**: Download table as CSV

### 2. Pitcher Detail Page

**Purpose**: Comprehensive view of individual pitcher with risk analysis and workload trends

#### Layout
```
┌─────────────────────────────────────────────────────────┐
│ Breadcrumb: Overview > Gerrit Cole                     │
├─────────────────────────────────────────────────────────┤
│ Pitcher Header                                          │
│ ┌─────────────────┐ ┌─────────────────┐                │
│ │ Name & Team     │ │ Current Risk    │                │
│ │ Gerrit Cole     │ │ Score: 23%      │                │
│ │ New York        │ │ Level: Medium   │                │
│ │ Yankees         │ │ Confidence: 85% │                │
│ └─────────────────┘ └─────────────────┘                │
│                                                         │
│ "Why Now" Panel (Risk Contributors)                     │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 🚨 High recent workload: 245 pitches in last 3     │ │
│ │    games → Consider +1 rest day                     │ │
│ │ 📉 Velocity decline: -2.1 MPH vs 30-day baseline   │ │
│ │    → Monitor in bullpen sessions                    │ │
│ │ ⏰ Short rest: Only 3 days since last appearance    │ │
│ │    → Consider skipping next start                   │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Charts Section                                          │
│ ┌─────────────────┐ ┌─────────────────┐                │
│ │ Workload        │ │ Velocity &      │                │
│ │ Trend           │ │ Spin Rate       │                │
│ │ (Pitch Counts)  │ │ Trends          │                │
│ └─────────────────┘ └─────────────────┘                │
│                                                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Risk Over Time (Sparkline)                          │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Recent Appearances Table                                │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Date | Pitches | Avg Vel | Avg Spin | Risk Score   │ │
│ │ -----|---------|---------|----------|-------------  │ │
│ │ 4/15 | 85      | 93.2    | 2350     | 23%          │ │
│ │ 4/10 | 95      | 94.1    | 2400     | 15%          │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Components

**Risk Contributors Panel**
```typescript
interface RiskContributor {
  name: string;
  value: number;
  direction: 'increasing' | 'decreasing';
  explanation: string;
  mitigation: string;
  severity: 'low' | 'medium' | 'high';
}
```

**Workload Chart**
```typescript
interface WorkloadDataPoint {
  date: string;
  pitchesThrown: number;
  avgVelocity: number;
  avgSpinRate: number;
  riskScore: number;
}
```

#### Interactions
- **Chart zoom**: Click and drag to zoom into time periods
- **Hover details**: Show exact values on chart hover
- **Date range selector**: Choose custom date ranges
- **Export data**: Download chart data as CSV
- **Share**: Generate shareable link

### 3. Team Overview Page

**Purpose**: Team-level risk assessment and pitcher management

#### Layout
```
┌─────────────────────────────────────────────────────────┐
│ Page Header: "New York Yankees - Pitcher Health"       │
├─────────────────────────────────────────────────────────┤
│ Team Summary Cards                                      │
│ ┌─────────────────┐ ┌─────────────────┐                │
│ │ Total Pitchers  │ │ High Risk       │                │
│ │ 15              │ │ 2               │                │
│ └─────────────────┘ └─────────────────┘                │
│ ┌─────────────────┐ ┌─────────────────┐                │
│ │ Avg Risk Score  │ │ Risk Trend      │                │
│ │ 18%             │ │ ↗️ Increasing   │                │
│ └─────────────────┘ └─────────────────┘                │
│                                                         │
│ Risk Distribution Chart                                 │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ [Low: 8] [Medium: 5] [High: 2]                     │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Pitchers Grid                                           │
│ ┌─────────────────┐ ┌─────────────────┐                │
│ │ Gerrit Cole     │ │ Carlos Rodón    │                │
│ │ Risk: Low       │ │ Risk: Medium    │                │
│ │ Trend: Stable   │ │ Trend: ↗️       │                │
│ │ [View Details]  │ │ [View Details]  │                │
│ └─────────────────┘ └─────────────────┘                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Component Library

### 1. Risk Badge Component

```typescript
interface RiskBadgeProps {
  level: 'low' | 'medium' | 'high';
  score: number;
  showPercentage?: boolean;
  size?: 'small' | 'medium' | 'large';
}
```

**Visual Design:**
- **Low**: Green background, white text, checkmark icon
- **Medium**: Yellow background, dark text, warning icon
- **High**: Red background, white text, alert icon

### 2. Trend Sparkline Component

```typescript
interface SparklineProps {
  data: number[];
  dates: string[];
  color: string;
  showValue?: boolean;
  height?: number;
}
```

**Features:**
- Smooth line chart with area fill
- Hover to show exact values
- Color-coded by trend direction
- Responsive to container size

### 3. Workload Chart Component

```typescript
interface WorkloadChartProps {
  data: WorkloadDataPoint[];
  showVelocity?: boolean;
  showSpinRate?: boolean;
  showRisk?: boolean;
  height?: number;
}
```

**Features:**
- Multi-line chart with dual y-axes
- Interactive legend
- Zoom and pan capabilities
- Export functionality

### 4. Risk Contributors Panel

```typescript
interface RiskContributorsProps {
  contributors: RiskContributor[];
  maxDisplay?: number;
  showMitigations?: boolean;
}
```

**Visual Design:**
- Card-based layout
- Color-coded by severity
- Expandable details
- Action buttons for mitigations

## Data Visualization Guidelines

### Chart Types

1. **Line Charts**: Time series data (velocity, spin rate, risk scores)
2. **Bar Charts**: Categorical data (pitch counts by appearance)
3. **Sparklines**: Trend indicators in tables
4. **Gauge Charts**: Risk scores and confidence levels
5. **Heatmaps**: Workload patterns over time

### Color Usage

- **Sequential**: Use single color with varying opacity for continuous data
- **Diverging**: Use red-yellow-green for risk levels
- **Categorical**: Use distinct colors for different pitchers/teams
- **Accessibility**: Ensure sufficient contrast ratios

### Interactivity

- **Hover**: Show detailed tooltips with exact values
- **Click**: Navigate to detailed views
- **Zoom**: Allow users to focus on specific time periods
- **Filter**: Enable data filtering and subsetting

## Accessibility Features

### Keyboard Navigation
- **Tab order**: Logical tab sequence through all interactive elements
- **Arrow keys**: Navigate through table rows and chart elements
- **Enter/Space**: Activate buttons and links
- **Escape**: Close modals and dropdowns

### Screen Reader Support
- **ARIA labels**: Descriptive labels for all interactive elements
- **Live regions**: Announce dynamic content updates
- **Landmarks**: Proper heading structure and navigation landmarks
- **Alt text**: Descriptive alt text for charts and images

### Visual Accessibility
- **Color contrast**: Minimum 4.5:1 ratio for normal text
- **Focus indicators**: Clear focus states for all interactive elements
- **Text sizing**: Support for browser text size adjustments
- **Motion**: Respect user's motion preferences

## Responsive Design

### Mobile Adaptations

**Staff Overview:**
- Stack summary cards vertically
- Convert table to card layout
- Collapse filters into dropdown
- Swipe gestures for navigation

**Pitcher Detail:**
- Stack charts vertically
- Full-width risk contributors
- Simplified navigation
- Touch-friendly interactions

**Team Overview:**
- Single-column pitcher grid
- Collapsible summary cards
- Simplified charts
- Mobile-optimized tables

### Tablet Adaptations

**Layout Adjustments:**
- Sidebar becomes collapsible
- Charts maintain aspect ratios
- Tables use horizontal scrolling
- Touch-friendly button sizes

## Performance Considerations

### Loading States
- **Skeleton screens**: Show content structure while loading
- **Progressive loading**: Load critical content first
- **Lazy loading**: Load charts and images on demand
- **Caching**: Cache API responses for better performance

### Data Optimization
- **Pagination**: Load data in chunks for large datasets
- **Debouncing**: Limit API calls for search and filters
- **Memoization**: Cache expensive calculations
- **Virtual scrolling**: For large tables

## Error Handling

### Error States
- **Network errors**: Show retry options with clear messaging
- **Data errors**: Display fallback content with error details
- **Validation errors**: Inline error messages with suggestions
- **Empty states**: Helpful messaging when no data available

### User Feedback
- **Loading indicators**: Show progress for long operations
- **Success messages**: Confirm successful actions
- **Warning messages**: Alert users to potential issues
- **Toast notifications**: Non-intrusive status updates

## Copy Guidelines

### Tone and Voice
- **Professional**: Use baseball terminology appropriately
- **Clear**: Avoid jargon, explain technical concepts
- **Actionable**: Every insight should suggest next steps
- **Confident**: Be definitive but not overconfident

### Key Phrases
- **Risk levels**: "Low risk", "Elevated risk", "High risk"
- **Trends**: "Risk increasing", "Risk decreasing", "Risk stable"
- **Actions**: "Consider monitoring", "Review workload", "Schedule rest"
- **Data quality**: "Limited data", "Sufficient data", "Complete data"

### Accessibility in Copy
- **Descriptive links**: "View Gerrit Cole's details" not "Click here"
- **Alt text**: Describe charts and data visualizations
- **Error messages**: Explain what went wrong and how to fix it
- **Success messages**: Confirm what action was completed
