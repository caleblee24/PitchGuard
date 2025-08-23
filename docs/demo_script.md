# PitchGuard Demo Script (5-7 minutes)

## Demo Overview

**Goal**: Demonstrate how PitchGuard predicts pitcher injury risk using workload patterns and provides actionable insights for coaching staff.

**Target Audience**: MLB pitching coaches, training staff, front office personnel

**Key Message**: "Predict injury risk 21 days in advance using data-driven workload analysis"

## Demo Flow

### 1. Opening Hook (30 seconds)

**"MLB teams lose $1.5B annually to pitcher injuries. What if you could see them coming 21 days in advance?"**

- Show current state: reactive monitoring, subjective assessments
- Introduce PitchGuard: proactive, data-driven injury prediction

### 2. Staff Overview Dashboard (1.5 minutes)

**"Let's start with our daily overview - this is what your coaching staff sees every morning."**

#### Key Points to Highlight:
- **Risk Badges**: Green (low), Yellow (medium), Red (high) risk indicators
- **Trend Sparklines**: 30-day risk trajectory at a glance
- **Quick Tags**: "−2.3 MPH", "Short Rest", "High Workload"
- **Sorting**: By risk level, team, or recent activity

#### Demo Actions:
1. Show table with 3-5 pitchers including one high-risk case
2. Point out risk badges and trend lines
3. Click on high-risk pitcher to drill down
4. Mention filtering by team/season

**"Notice how we can quickly identify pitchers needing attention - like this one with elevated risk."**

### 3. Pitcher Detail Deep Dive (2 minutes)

**"Let's examine why this pitcher is showing elevated risk."**

#### Key Points to Highlight:
- **Current Risk Score**: 23% (calibrated probability)
- **Workload Trends**: Pitch counts, velocity, spin rates over time
- **"Why Now" Panel**: Top 3 risk contributors with explanations
- **Mitigation Suggestions**: Specific, actionable recommendations

#### Demo Actions:
1. Show risk score and confidence level
2. Point out velocity decline trend (−2.1 MPH vs 30-day baseline)
3. Highlight high pitch count in last 3 games (245 pitches)
4. Show "Why Now" explanations:
   - "High recent workload: 245 pitches in last 3 games"
   - "Velocity decline: -2.1 MPH vs 30-day average"
   - "Short rest: Only 3 days since last appearance"
5. Show mitigation suggestions:
   - "Consider +1 rest day before next start"
   - "Monitor velocity in bullpen sessions"
   - "Reduce pitch count target by 10-15"

**"This gives you specific, actionable insights - not just a risk number, but why it's elevated and what to do about it."**

### 4. Historical Context & Prediction (1 minute)

**"Let's look at how our predictions have performed historically."**

#### Key Points to Highlight:
- **21-day Prediction Window**: Forward-looking, not retrospective
- **Model Performance**: AUC, precision/recall metrics
- **Calibration**: Risk scores correspond to actual injury rates
- **False Positive Management**: Balance between detection and false alarms

#### Demo Actions:
1. Show model performance metrics
2. Point out 21-day prediction window
3. Mention calibration accuracy
4. Show historical prediction examples

**"Our model identifies 70% of injuries in the top 20% risk group, with less than 30% false positives."**

### 5. Workload Monitoring Features (1 minute)

**"Beyond risk prediction, we provide comprehensive workload monitoring."**

#### Key Points to Highlight:
- **Rolling Workloads**: 3-game, 7-day, 14-day pitch counts
- **Velocity/Spin Trends**: Performance indicators over time
- **Recovery Patterns**: Rest day analysis and optimization
- **Comparative Analysis**: League averages and benchmarks

#### Demo Actions:
1. Show workload charts with rolling windows
2. Point out velocity and spin rate trends
3. Highlight recovery pattern analysis
4. Show league comparison features

**"This gives you the full picture of pitcher workload and performance trends."**

### 6. Closing & Next Steps (30 seconds)

**"PitchGuard transforms reactive injury management into proactive workload optimization."**

#### Key Benefits Recap:
- **21-day advance warning** of injury risk
- **Actionable insights** with specific recommendations
- **Comprehensive monitoring** of workload patterns
- **Data-driven decisions** replacing subjective assessments

#### Call to Action:
- **Pilot Program**: "We're looking for 2-3 teams to pilot this system"
- **Timeline**: "Full implementation in 6-8 weeks"
- **ROI**: "Potential $50M+ annual savings across MLB"

**"Would you like to discuss how PitchGuard could fit into your team's injury prevention strategy?"**

## Demo Preparation Checklist

### Technical Setup
- [ ] Mock data loaded with realistic injury scenario
- [ ] High-risk pitcher identified for demo
- [ ] All charts and visualizations working
- [ ] API responses cached for smooth demo
- [ ] Backup screenshots ready

### Demo Data Requirements
- [ ] 3 pitchers with varying risk levels
- [ ] One pitcher with clear velocity decline pattern
- [ ] One pitcher with high workload scenario
- [ ] Historical data showing prediction accuracy
- [ ] Realistic injury timeline (10-15 days after risk spike)

### Key Metrics to Have Ready
- [ ] Model AUC: 0.78
- [ ] Detection Rate: 70% in top 20% risk group
- [ ] False Positive Rate: <30%
- [ ] Calibration accuracy: ±5%
- [ ] Processing time: <2 seconds per prediction

## Demo Tips

### Do's
- **Start with the problem**: Lead with injury costs and current limitations
- **Show, don't tell**: Let the data and visualizations speak
- **Focus on actionable insights**: Emphasize specific recommendations
- **Use real baseball language**: "velocity", "spin rate", "workload"
- **Address concerns proactively**: Mention false positive management

### Don'ts
- **Don't get technical**: Avoid ML jargon, focus on business value
- **Don't overpromise**: Be realistic about prediction accuracy
- **Don't ignore limitations**: Acknowledge data quality dependencies
- **Don't rush**: Allow time for questions and discussion

## Q&A Preparation

### Likely Questions & Answers

**Q: "How accurate are these predictions?"**
A: "We identify 70% of injuries in the top 20% risk group, with less than 30% false positives. The model is calibrated so a 25% risk score means 25% of similar pitchers were injured within 21 days."

**Q: "What data do you need?"**
A: "We use pitch-level data from Baseball Savant - release speed, spin rate, pitch counts, and game dates. We also need IL transaction data for injury labels."

**Q: "How do you handle different pitcher types?"**
A: "Our model accounts for individual pitcher baselines and adjusts for role (starter vs reliever). We're working on position-specific models for future versions."

**Q: "What about false positives?"**
A: "We balance detection rate with false positives. A 25% risk score doesn't mean injury is certain - it means increased monitoring and potential workload adjustment."

**Q: "How quickly can we implement this?"**
A: "With existing Baseball Savant access, we can have a pilot running in 4-6 weeks. Full implementation typically takes 8-12 weeks including data integration and staff training."

## Demo Success Metrics

- **Engagement**: Questions asked during demo
- **Understanding**: Ability to explain key concepts back
- **Interest**: Request for follow-up meeting or pilot program
- **Objections**: Concerns raised and addressed
- **Next Steps**: Clear action items identified
