# UI Copy Guidelines

## Overview

This document defines all text content, messaging, and copy guidelines for the PitchGuard frontend interface. The copy prioritizes clarity, actionable insights, and professional tone appropriate for MLB coaching staff.

## Design Principles

1. **Clear and Concise**: Use simple, direct language
2. **Actionable**: Every insight should suggest next steps
3. **Professional**: Appropriate for baseball professionals
4. **Confident but Cautious**: Avoid overconfident predictions
5. **Accessible**: Use plain language, avoid jargon

## Page Headers and Navigation

### Main Navigation
- **Overview**: "Pitcher Overview"
- **Teams**: "Team Management"
- **Analytics**: "Advanced Analytics"
- **Settings**: "System Settings"

### Page Titles
- **Dashboard**: "Pitcher Health Dashboard"
- **Pitcher Detail**: "{Pitcher Name} - Risk Assessment"
- **Team View**: "{Team Name} - Pitcher Health"
- **Analytics**: "Performance Analytics"

## Risk Assessment Copy

### Risk Levels

**Low Risk (0-20%)**
- **Badge**: "Low Risk"
- **Description**: "Normal workload patterns, no immediate concerns"
- **Action**: "Continue monitoring"

**Medium Risk (21-50%)**
- **Badge**: "Elevated Risk"
- **Description**: "Some concerning patterns detected"
- **Action**: "Review workload and consider adjustments"

**High Risk (51-100%)**
- **Badge**: "High Risk"
- **Description**: "Multiple risk factors present"
- **Action**: "Immediate attention recommended"

### Risk Contributors

**High Workload**
- **Title**: "High Recent Workload"
- **Description**: "Pitcher threw {X} pitches in last {Y} games"
- **Mitigation**: "Consider +1 rest day before next start"

**Velocity Decline**
- **Title**: "Velocity Decline"
- **Description**: "Average velocity down {X} MPH vs 30-day baseline"
- **Mitigation**: "Monitor velocity in bullpen sessions"

**Short Rest**
- **Title**: "Short Rest Period"
- **Description**: "Only {X} days rest since last appearance"
- **Mitigation**: "Consider skipping next start"

**Spin Rate Drop**
- **Title**: "Spin Rate Decline"
- **Description**: "Average spin rate down {X} RPM vs 30-day baseline"
- **Mitigation**: "Review pitch mechanics and grip"

**Consecutive High Workload**
- **Title**: "Consecutive High Workload"
- **Description**: "{X} consecutive games with 90+ pitches"
- **Mitigation**: "Schedule additional rest or bullpen session"

## Dashboard Copy

### Summary Cards

**Total Pitchers**
- **Title**: "Active Pitchers"
- **Subtitle**: "Currently monitored"

**High Risk Count**
- **Title**: "High Risk"
- **Subtitle**: "Require attention"

**Average Risk**
- **Title**: "Avg Risk Score"
- **Subtitle**: "Team average"

**Recent Injuries**
- **Title**: "Recent Injuries"
- **Subtitle**: "Last 30 days"

### Table Headers

**Pitchers Table**
- **Name**: "Pitcher"
- **Team**: "Team"
- **Last Appearance**: "Last App"
- **Risk Level**: "Risk"
- **Trend**: "Trend"
- **Actions**: "Actions"

**Workload Table**
- **Date**: "Date"
- **Pitches**: "Pitches"
- **Velocity**: "Avg Vel"
- **Spin Rate**: "Avg Spin"
- **Risk Score**: "Risk"

### Filter Labels

**Team Filter**
- **Label**: "Team"
- **Placeholder**: "Select team"
- **All Teams**: "All Teams"

**Season Filter**
- **Label**: "Season"
- **Placeholder**: "Select season"
- **Current**: "Current Season"

**Risk Level Filter**
- **Label**: "Risk Level"
- **Options**: "All", "Low", "Medium", "High"

**Sort Options**
- **Label**: "Sort by"
- **Options**: "Risk (High to Low)", "Risk (Low to High)", "Name", "Last Appearance"

## Pitcher Detail Copy

### Header Information

**Pitcher Name**
- **Format**: "{First Name} {Last Name}"
- **Team**: "{Team Name}"

**Current Risk Assessment**
- **Title**: "Current Risk Assessment"
- **Score**: "{X}% risk"
- **Level**: "{Risk Level}"
- **Confidence**: "{X}% confidence"

**Last Appearance**
- **Label**: "Last Appearance"
- **Format**: "{Date} ({X} days ago)"

### Why Now Panel

**Panel Title**: "Why Elevated Risk?"

**Contributor Format**
- **Icon**: 🚨 (High), 📉 (Decline), ⏰ (Time), 📊 (Pattern)
- **Title**: "{Risk Factor Name}"
- **Value**: "{Numeric Value}"
- **Description**: "{Plain English explanation}"
- **Mitigation**: "→ {Actionable recommendation}"

**Examples**:
- "🚨 High recent workload: 245 pitches in last 3 games → Consider +1 rest day"
- "📉 Velocity decline: -2.1 MPH vs 30-day baseline → Monitor in bullpen sessions"
- "⏰ Short rest: Only 3 days since last appearance → Consider skipping next start"

### Workload Summary

**Section Title**: "Workload Summary"

**Metrics**:
- **Last 3 Games**: "{X} pitches"
- **Last 7 Days**: "{X} pitches"
- **Avg Velocity**: "{X.X} MPH"
- **Velocity Trend**: "{+/-X.X} MPH vs 30-day"
- **Rest Days**: "{X} days"

### Charts and Visualizations

**Workload Trend Chart**
- **Title**: "Pitch Count Trend"
- **Y-Axis**: "Pitches"
- **X-Axis**: "Date"
- **Legend**: "Pitches per Appearance"

**Velocity Trend Chart**
- **Title**: "Velocity & Spin Rate Trends"
- **Y-Axis**: "MPH / RPM"
- **X-Axis**: "Date"
- **Legend**: "Avg Velocity", "Avg Spin Rate", "30-day Baseline"

**Risk Trend Chart**
- **Title**: "Risk Over Time"
- **Y-Axis**: "Risk %"
- **X-Axis**: "Date"
- **Legend**: "Risk Score"

## Team Overview Copy

### Team Summary

**Team Header**
- **Format**: "{Team Name} - Pitcher Health"
- **Subtitle**: "Risk assessment for all active pitchers"

**Summary Cards**
- **Total Pitchers**: "{X} Active Pitchers"
- **High Risk**: "{X} High Risk"
- **Medium Risk**: "{X} Elevated Risk"
- **Low Risk**: "{X} Low Risk"
- **Average Risk**: "{X}% Team Average"

### Risk Distribution

**Chart Title**: "Risk Distribution"
**Legend**:
- **Low**: "Low Risk ({X})"
- **Medium**: "Elevated Risk ({X})"
- **High**: "High Risk ({X})"

### Alerts Panel

**Panel Title**: "Active Alerts"

**Alert Format**:
- **High Risk Alert**: "🚨 {X} pitchers showing elevated injury risk"
- **Trend Alert**: "📈 Team risk trending upward"
- **Workload Alert**: "⚾ High workload detected across {X} pitchers"

## Error Messages

### API Errors

**Network Error**
- **Title**: "Connection Error"
- **Message**: "Unable to connect to server. Please check your internet connection and try again."
- **Action**: "Retry"

**Server Error**
- **Title**: "Server Error"
- **Message**: "Something went wrong on our end. Please try again in a few minutes."
- **Action**: "Retry"

**Data Not Found**
- **Title**: "Data Not Available"
- **Message**: "No data available for the selected criteria. Try adjusting your filters or date range."
- **Action**: "Adjust Filters"

### Validation Errors

**Invalid Date**
- **Message**: "Please select a valid date within the last 30 days."

**Invalid Pitcher**
- **Message**: "Pitcher not found. Please select a different pitcher from the list."

**Date Range Too Large**
- **Message**: "Date range cannot exceed 365 days. Please select a smaller range."

## Loading States

### Page Loading
- **Message**: "Loading pitcher data..."
- **Subtitle**: "Please wait while we fetch the latest information"

### Data Processing
- **Message**: "Processing data..."
- **Subtitle**: "Analyzing workload patterns and risk factors"

### Chart Loading
- **Message**: "Generating charts..."
- **Subtitle**: "Creating visualizations of workload trends"

## Empty States

### No Data Available
- **Title**: "No Data Available"
- **Message**: "No pitcher data found for the selected criteria."
- **Action**: "Try adjusting your filters or date range"

### No High Risk Pitchers
- **Title**: "All Clear"
- **Message**: "No pitchers currently showing elevated risk levels."
- **Subtitle**: "Continue monitoring for any changes"

### No Recent Activity
- **Title**: "No Recent Activity"
- **Message**: "No recent appearances recorded for this pitcher."
- **Subtitle**: "Data will appear once new games are played"

## Success Messages

### Data Updated
- **Message**: "Data updated successfully"
- **Duration**: 3 seconds

### Settings Saved
- **Message**: "Settings saved successfully"
- **Duration**: 3 seconds

### Export Complete
- **Message**: "Data exported successfully"
- **Duration**: 3 seconds

## Tooltips and Help Text

### Risk Score Tooltip
- **Text**: "Risk score represents the probability of injury within the next 21 days based on current workload patterns and performance trends."

### Confidence Tooltip
- **Text**: "Confidence level indicates how reliable this risk assessment is based on available data quality and completeness."

### Trend Tooltip
- **Text**: "Trend shows whether risk is increasing (↗️), decreasing (↘️), or stable (→) over the past 30 days."

### Data Completeness Tooltip
- **Text**: "Data completeness indicates how much recent data is available for this pitcher. Higher completeness means more reliable assessments."

## Accessibility Text

### Chart Descriptions
- **Workload Chart**: "Line chart showing pitch count trend over time with dates on x-axis and pitch counts on y-axis"
- **Velocity Chart**: "Multi-line chart showing velocity and spin rate trends with 30-day baseline reference line"
- **Risk Chart**: "Area chart showing risk score progression over time with risk percentage on y-axis"

### Button Descriptions
- **View Details**: "View detailed risk assessment for {pitcher name}"
- **Export Data**: "Export current data to CSV file"
- **Filter Results**: "Filter pitchers by team, risk level, or other criteria"
- **Sort Table**: "Sort table by {column name} in {ascending/descending} order"

### Navigation Descriptions
- **Back to Overview**: "Return to pitcher overview dashboard"
- **Next Pitcher**: "View next pitcher in list"
- **Previous Pitcher**: "View previous pitcher in list"

## Mobile-Specific Copy

### Responsive Labels
- **Desktop**: "Last Appearance"
- **Mobile**: "Last App"

- **Desktop**: "Average Velocity"
- **Mobile**: "Avg Vel"

- **Desktop**: "Risk Assessment"
- **Mobile**: "Risk"

### Touch Actions
- **Swipe**: "Swipe left/right to view different time periods"
- **Tap**: "Tap pitcher row to view details"
- **Long Press**: "Long press to access quick actions"

## Seasonal Copy

### Pre-Season
- **Message**: "Pre-season data limited. Risk assessments will become more accurate as regular season begins."

### Regular Season
- **Message**: "Regular season monitoring active. Daily updates provide real-time risk assessments."

### Post-Season
- **Message**: "Post-season monitoring. Reduced workload may affect risk assessment accuracy."

### Off-Season
- **Message**: "Off-season mode. Historical data available for analysis and planning."

## Contextual Messages

### First-Time User
- **Welcome**: "Welcome to PitchGuard! This dashboard helps you monitor pitcher workload and injury risk."
- **Tutorial**: "Click on any pitcher to view detailed risk assessment and recommendations."

### Data Quality Warnings
- **Low Data**: "Limited data available for this pitcher. Risk assessment may be less accurate."
- **Missing Data**: "Some data missing for recent appearances. Consider manual verification."

### System Maintenance
- **Scheduled**: "System maintenance scheduled for {time}. Some features may be temporarily unavailable."
- **Unplanned**: "System maintenance in progress. Please try again in a few minutes."

## Professional Tone Guidelines

### Do's
- Use clear, professional language
- Provide specific, actionable recommendations
- Acknowledge uncertainty when appropriate
- Use baseball terminology correctly
- Maintain consistent terminology

### Don'ts
- Avoid overly technical jargon
- Don't make definitive injury predictions
- Don't use casual or informal language
- Don't oversell system capabilities
- Don't ignore data limitations

### Tone Examples

**Good**: "Consider monitoring velocity in bullpen sessions"
**Avoid**: "You should definitely check his velocity"

**Good**: "Risk assessment suggests elevated concern"
**Avoid**: "This pitcher is definitely going to get hurt"

**Good**: "Limited data available for recent period"
**Avoid**: "We don't have enough info to tell you anything"

## Localization Considerations

### Date Formats
- **US**: "April 15, 2024"
- **International**: "15 April 2024"

### Number Formats
- **US**: "1,234.5"
- **International**: "1 234,5"

### Units
- **Velocity**: "MPH" (US), "km/h" (International)
- **Distance**: "feet" (US), "meters" (International)

### Team Names
- **US**: Full team names
- **International**: Abbreviated team names with country indicators
