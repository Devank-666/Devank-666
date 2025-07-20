import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Meta Ads Performance Projections",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .ai-suggestion {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00bcd4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

class MetaAdsProjector:
    def __init__(self):
        self.initialize_sample_data()
        
    def initialize_sample_data(self):
        """Generate realistic Meta Ads sample data"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        # Generate realistic ad metrics
        data = []
        for i, date in enumerate(dates):
            # Simulate seasonal trends and day-of-week effects
            day_of_week = date.weekday()
            is_weekend = day_of_week >= 5
            month = date.month
            
            # Base metrics with realistic relationships
            impressions = np.random.normal(50000, 15000) * (0.8 if is_weekend else 1.0) * (1.2 if month in [11, 12] else 1.0)
            impressions = max(10000, impressions)
            
            ctr = np.random.normal(1.8, 0.5) if not is_weekend else np.random.normal(1.4, 0.4)
            ctr = max(0.5, min(5.0, ctr))
            
            clicks = impressions * (ctr / 100)
            
            cpc = np.random.normal(1.25, 0.3) * (1.1 if is_weekend else 1.0)
            cpc = max(0.5, min(3.0, cpc))
            
            spend = clicks * cpc
            
            conversion_rate = np.random.normal(3.2, 0.8) if not is_weekend else np.random.normal(2.8, 0.7)
            conversion_rate = max(1.0, min(8.0, conversion_rate))
            
            conversions = clicks * (conversion_rate / 100)
            
            revenue_per_conversion = np.random.normal(85, 20)
            revenue_per_conversion = max(30, revenue_per_conversion)
            
            revenue = conversions * revenue_per_conversion
            
            data.append({
                'date': date,
                'impressions': int(impressions),
                'clicks': int(clicks),
                'spend': round(spend, 2),
                'conversions': int(conversions),
                'revenue': round(revenue, 2),
                'ctr': round(ctr, 2),
                'cpc': round(cpc, 2),
                'conversion_rate': round(conversion_rate, 2),
                'roas': round(revenue / spend if spend > 0 else 0, 2),
                'cpa': round(spend / conversions if conversions > 0 else 0, 2)
            })
        
        self.historical_data = pd.DataFrame(data)
        
    def calculate_projections(self, days_to_project=30, budget_change=0, target_improvements=None):
        """Calculate future projections based on historical data"""
        if target_improvements is None:
            target_improvements = {}
            
        # Prepare features for ML model
        self.historical_data['day_of_week'] = self.historical_data['date'].dt.dayofweek
        self.historical_data['month'] = self.historical_data['date'].dt.month
        self.historical_data['day_of_year'] = self.historical_data['date'].dt.dayofyear
        
        # Features for prediction
        features = ['day_of_week', 'month', 'day_of_year']
        X = self.historical_data[features]
        
        # Train models for key metrics
        models = {}
        metrics_to_predict = ['impressions', 'ctr', 'cpc', 'conversion_rate']
        
        for metric in metrics_to_predict:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, self.historical_data[metric])
            models[metric] = model
        
        # Generate future dates
        last_date = self.historical_data['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_project, freq='D')
        
        # Create future features
        future_data = []
        for date in future_dates:
            future_features = {
                'date': date,
                'day_of_week': date.weekday(),
                'month': date.month,
                'day_of_year': date.dayofyear
            }
            future_data.append(future_features)
        
        future_df = pd.DataFrame(future_data)
        X_future = future_df[features]
        
        # Predict future metrics
        predictions = {}
        for metric in metrics_to_predict:
            predictions[metric] = models[metric].predict(X_future)
        
        # Apply improvements and budget changes
        budget_multiplier = 1 + (budget_change / 100)
        
        projected_data = []
        for i, date in enumerate(future_dates):
            impressions = predictions['impressions'][i] * budget_multiplier
            ctr = predictions['ctr'][i] * (1 + target_improvements.get('ctr_improvement', 0) / 100)
            cpc = predictions['cpc'][i] * (1 + target_improvements.get('cpc_change', 0) / 100)
            conversion_rate = predictions['conversion_rate'][i] * (1 + target_improvements.get('conversion_rate_improvement', 0) / 100)
            
            clicks = impressions * (ctr / 100)
            spend = clicks * cpc
            conversions = clicks * (conversion_rate / 100)
            
            # Estimate revenue based on historical average
            avg_revenue_per_conversion = self.historical_data['revenue'].sum() / self.historical_data['conversions'].sum()
            revenue = conversions * avg_revenue_per_conversion * (1 + target_improvements.get('aov_improvement', 0) / 100)
            
            projected_data.append({
                'date': date,
                'impressions': int(impressions),
                'clicks': int(clicks),
                'spend': round(spend, 2),
                'conversions': int(conversions),
                'revenue': round(revenue, 2),
                'ctr': round(ctr, 2),
                'cpc': round(cpc, 2),
                'conversion_rate': round(conversion_rate, 2),
                'roas': round(revenue / spend if spend > 0 else 0, 2),
                'cpa': round(spend / conversions if conversions > 0 else 0, 2)
            })
        
        return pd.DataFrame(projected_data)

class AIAssistant:
    def __init__(self):
        self.recommendations = []
        
    def analyze_performance(self, historical_data, projected_data=None):
        """Analyze performance and provide AI-driven recommendations"""
        
        # Calculate recent performance metrics
        recent_data = historical_data.tail(30)  # Last 30 days
        avg_roas = recent_data['roas'].mean()
        avg_ctr = recent_data['ctr'].mean()
        avg_cpc = recent_data['cpc'].mean()
        avg_conversion_rate = recent_data['conversion_rate'].mean()
        avg_cpa = recent_data['cpa'].mean()
        
        recommendations = []
        
        # ROAS Analysis
        if avg_roas < 3.0:
            recommendations.append({
                'type': 'critical',
                'metric': 'ROAS',
                'current': avg_roas,
                'target': 4.0,
                'priority': 'High',
                'recommendation': 'Your ROAS is below the recommended 3:1 ratio. Focus on improving conversion rates and average order value.',
                'actions': [
                    'Review and optimize ad creative for better quality scores',
                    'Implement audience segmentation for better targeting',
                    'Test landing page optimization for higher conversion rates',
                    'Consider increasing product prices or upselling strategies'
                ]
            })
        elif avg_roas < 4.0:
            recommendations.append({
                'type': 'warning',
                'metric': 'ROAS',
                'current': avg_roas,
                'target': 5.0,
                'priority': 'Medium',
                'recommendation': 'Good ROAS but room for improvement. Focus on scaling successful campaigns.',
                'actions': [
                    'Identify top-performing ad sets and increase budgets',
                    'Expand successful audiences with lookalike audiences',
                    'Test new creative formats and messaging'
                ]
            })
        else:
            recommendations.append({
                'type': 'success',
                'metric': 'ROAS',
                'current': avg_roas,
                'target': avg_roas * 1.1,
                'priority': 'Low',
                'recommendation': 'Excellent ROAS! Focus on scaling and maintaining performance.',
                'actions': [
                    'Gradually increase budgets while monitoring performance',
                    'Expand to new audiences and platforms',
                    'Implement automated bidding strategies'
                ]
            })
        
        # CTR Analysis
        if avg_ctr < 1.0:
            recommendations.append({
                'type': 'critical',
                'metric': 'CTR',
                'current': avg_ctr,
                'target': 2.0,
                'priority': 'High',
                'recommendation': 'CTR is significantly below industry average. Ad relevance needs immediate attention.',
                'actions': [
                    'Refresh ad creative with more compelling visuals',
                    'Test new ad copy with stronger calls-to-action',
                    'Review audience targeting for better relevance',
                    'Implement dynamic product ads if applicable'
                ]
            })
        elif avg_ctr < 1.5:
            recommendations.append({
                'type': 'warning',
                'metric': 'CTR',
                'current': avg_ctr,
                'target': 2.0,
                'priority': 'Medium',
                'recommendation': 'CTR is below optimal. Consider creative refresh and audience refinement.',
                'actions': [
                    'A/B test new creative concepts',
                    'Optimize audience targeting',
                    'Test different ad formats (video, carousel, etc.)'
                ]
            })
        
        # CPC Analysis
        if avg_cpc > 2.0:
            recommendations.append({
                'type': 'warning',
                'metric': 'CPC',
                'current': avg_cpc,
                'target': 1.5,
                'priority': 'Medium',
                'recommendation': 'CPC is higher than optimal. Focus on improving quality score and audience targeting.',
                'actions': [
                    'Improve ad relevance score through better targeting',
                    'Optimize landing page experience',
                    'Test manual bidding strategies',
                    'Exclude poor-performing placements and audiences'
                ]
            })
        
        # Conversion Rate Analysis
        if avg_conversion_rate < 2.0:
            recommendations.append({
                'type': 'critical',
                'metric': 'Conversion Rate',
                'current': avg_conversion_rate,
                'target': 3.5,
                'priority': 'High',
                'recommendation': 'Conversion rate is below industry standards. Landing page optimization is crucial.',
                'actions': [
                    'Conduct landing page A/B tests',
                    'Improve page loading speed',
                    'Optimize checkout process',
                    'Implement trust signals and social proof',
                    'Review traffic quality and audience targeting'
                ]
            })
        elif avg_conversion_rate < 3.0:
            recommendations.append({
                'type': 'warning',
                'metric': 'Conversion Rate',
                'current': avg_conversion_rate,
                'target': 4.0,
                'priority': 'Medium',
                'recommendation': 'Conversion rate has room for improvement. Focus on user experience optimization.',
                'actions': [
                    'Test different landing page layouts',
                    'Optimize mobile experience',
                    'Implement exit-intent popups',
                    'Add customer reviews and testimonials'
                ]
            })
        
        return recommendations
    
    def get_scaling_recommendations(self, current_budget, current_roas, target_roas=None):
        """Provide budget scaling recommendations"""
        scaling_recs = []
        
        if current_roas >= 4.0:
            scaling_recs.append({
                'action': 'Aggressive Scaling',
                'budget_increase': '50-100%',
                'rationale': 'High ROAS allows for aggressive scaling',
                'monitoring': 'Monitor ROAS closely and scale back if it drops below 3.5'
            })
        elif current_roas >= 3.0:
            scaling_recs.append({
                'action': 'Conservative Scaling',
                'budget_increase': '20-30%',
                'rationale': 'Moderate ROAS suggests careful scaling approach',
                'monitoring': 'Increase budget gradually while maintaining current performance'
            })
        else:
            scaling_recs.append({
                'action': 'Optimization Before Scaling',
                'budget_increase': '0%',
                'rationale': 'Focus on improving ROAS before increasing budget',
                'monitoring': 'Implement optimization strategies first'
            })
        
        return scaling_recs

# Initialize classes
@st.cache_resource
def initialize_projector():
    return MetaAdsProjector()

@st.cache_resource 
def initialize_ai_assistant():
    return AIAssistant()

projector = initialize_projector()
ai_assistant = initialize_ai_assistant()

# Main App
def main():
    st.markdown('<h1 class="main-header">📊 Meta Ads Performance Projections & AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🎯 Projection Settings")
    
    # Projection parameters
    projection_days = st.sidebar.slider("Days to Project", min_value=7, max_value=90, value=30)
    budget_change = st.sidebar.slider("Budget Change (%)", min_value=-50, max_value=100, value=0)
    
    st.sidebar.subheader("🎯 Target Improvements")
    ctr_improvement = st.sidebar.slider("CTR Improvement (%)", min_value=0, max_value=50, value=0)
    conversion_rate_improvement = st.sidebar.slider("Conversion Rate Improvement (%)", min_value=0, max_value=50, value=0)
    cpc_change = st.sidebar.slider("CPC Change (%)", min_value=-30, max_value=30, value=0)
    aov_improvement = st.sidebar.slider("AOV Improvement (%)", min_value=0, max_value=50, value=0)
    
    target_improvements = {
        'ctr_improvement': ctr_improvement,
        'conversion_rate_improvement': conversion_rate_improvement,
        'cpc_change': cpc_change,
        'aov_improvement': aov_improvement
    }
    
    # Calculate projections
    projected_data = projector.calculate_projections(
        days_to_project=projection_days,
        budget_change=budget_change,
        target_improvements=target_improvements
    )
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Dashboard", "🔮 Projections", "🤖 AI Insights", "📊 Historical Analysis", "⚙️ Scenario Planning"])
    
    with tab1:
        show_dashboard(projector.historical_data, projected_data)
    
    with tab2:
        show_projections(projector.historical_data, projected_data)
    
    with tab3:
        show_ai_insights(projector.historical_data, projected_data, ai_assistant)
    
    with tab4:
        show_historical_analysis(projector.historical_data)
    
    with tab5:
        show_scenario_planning(projector, target_improvements)

def show_dashboard(historical_data, projected_data):
    """Display main KPI dashboard"""
    st.header("📈 Performance Dashboard")
    
    # Recent performance metrics
    recent_data = historical_data.tail(30)
    projected_totals = projected_data.sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Spend (30d)",
            value=f"${recent_data['spend'].sum():,.2f}",
            delta=f"${projected_totals['spend'] - recent_data['spend'].sum():,.2f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Revenue (30d)",
            value=f"${recent_data['revenue'].sum():,.2f}",
            delta=f"${projected_totals['revenue'] - recent_data['revenue'].sum():,.2f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_roas = recent_data['roas'].mean()
        projected_roas = projected_data['roas'].mean()
        st.metric(
            label="Average ROAS",
            value=f"{current_roas:.2f}",
            delta=f"{projected_roas - current_roas:.2f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_cpa = recent_data['cpa'].mean()
        projected_cpa = projected_data['cpa'].mean()
        st.metric(
            label="Average CPA",
            value=f"${current_cpa:.2f}",
            delta=f"${projected_cpa - current_cpa:.2f}",
            delta_color="inverse"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance trend charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue vs Spend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data['date'],
            y=recent_data['revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=recent_data['date'],
            y=recent_data['spend'],
            mode='lines+markers',
            name='Spend',
            line=dict(color='red')
        ))
        fig.update_layout(title="Revenue vs Spend (Last 30 Days)", xaxis_title="Date", yaxis_title="Amount ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROAS trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data['date'],
            y=recent_data['roas'],
            mode='lines+markers',
            name='ROAS',
            line=dict(color='blue')
        ))
        fig.add_hline(y=3.0, line_dash="dash", line_color="orange", annotation_text="Target ROAS: 3.0")
        fig.update_layout(title="ROAS Trend (Last 30 Days)", xaxis_title="Date", yaxis_title="ROAS")
        st.plotly_chart(fig, use_container_width=True)

def show_projections(historical_data, projected_data):
    """Display projection charts and analysis"""
    st.header("🔮 Future Projections")
    
    # Combine historical and projected data for visualization
    combined_data = pd.concat([
        historical_data.tail(30).assign(type='Historical'),
        projected_data.assign(type='Projected')
    ])
    
    # Revenue projection chart
    fig = px.line(combined_data, x='date', y='revenue', color='type',
                  title="Revenue Projection", labels={'revenue': 'Revenue ($)', 'date': 'Date'})
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Spend projection
        fig = px.line(combined_data, x='date', y='spend', color='type',
                      title="Spend Projection", labels={'spend': 'Spend ($)', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROAS projection
        fig = px.line(combined_data, x='date', y='roas', color='type',
                      title="ROAS Projection", labels={'roas': 'ROAS', 'date': 'Date'})
        fig.add_hline(y=3.0, line_dash="dash", line_color="red", annotation_text="Target: 3.0")
        st.plotly_chart(fig, use_container_width=True)
    
    # Projection summary
    st.subheader("📊 Projection Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💰 Financial Projections")
        st.write(f"**Total Projected Spend:** ${projected_data['spend'].sum():,.2f}")
        st.write(f"**Total Projected Revenue:** ${projected_data['revenue'].sum():,.2f}")
        st.write(f"**Total Projected Profit:** ${projected_data['revenue'].sum() - projected_data['spend'].sum():,.2f}")
        
    with col2:
        st.markdown("### 📈 Performance Metrics")
        st.write(f"**Average Projected ROAS:** {projected_data['roas'].mean():.2f}")
        st.write(f"**Average Projected CPA:** ${projected_data['cpa'].mean():.2f}")
        st.write(f"**Average Projected CTR:** {projected_data['ctr'].mean():.2f}%")
        
    with col3:
        st.markdown("### 🎯 Volume Metrics")
        st.write(f"**Total Projected Impressions:** {projected_data['impressions'].sum():,.0f}")
        st.write(f"**Total Projected Clicks:** {projected_data['clicks'].sum():,.0f}")
        st.write(f"**Total Projected Conversions:** {projected_data['conversions'].sum():,.0f}")

def show_ai_insights(historical_data, projected_data, ai_assistant):
    """Display AI-driven insights and recommendations"""
    st.header("🤖 AI Performance Insights")
    
    # Get AI recommendations
    recommendations = ai_assistant.analyze_performance(historical_data, projected_data)
    
    # Display recommendations by priority
    high_priority = [r for r in recommendations if r['priority'] == 'High']
    medium_priority = [r for r in recommendations if r['priority'] == 'Medium']
    low_priority = [r for r in recommendations if r['priority'] == 'Low']
    
    if high_priority:
        st.markdown("### 🚨 High Priority Issues")
        for rec in high_priority:
            if rec['type'] == 'critical':
                st.markdown(f'<div class="warning-box">', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-suggestion">', unsafe_allow_html=True)
            
            st.markdown(f"**{rec['metric']} - Current: {rec['current']:.2f} | Target: {rec['target']:.2f}**")
            st.write(rec['recommendation'])
            st.markdown("**Action Items:**")
            for action in rec['actions']:
                st.write(f"• {action}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    if medium_priority:
        st.markdown("### ⚠️ Medium Priority Optimizations")
        for rec in medium_priority:
            st.markdown(f'<div class="ai-suggestion">', unsafe_allow_html=True)
            st.markdown(f"**{rec['metric']} - Current: {rec['current']:.2f} | Target: {rec['target']:.2f}**")
            st.write(rec['recommendation'])
            st.markdown("**Action Items:**")
            for action in rec['actions']:
                st.write(f"• {action}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    if low_priority:
        st.markdown("### ✅ Optimization Opportunities")
        for rec in low_priority:
            st.markdown(f'<div class="success-box">', unsafe_allow_html=True)
            st.markdown(f"**{rec['metric']} - Current: {rec['current']:.2f} | Target: {rec['target']:.2f}**")
            st.write(rec['recommendation'])
            st.markdown("**Action Items:**")
            for action in rec['actions']:
                st.write(f"• {action}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Budget scaling recommendations
    st.markdown("### 💰 Budget Scaling Recommendations")
    recent_data = historical_data.tail(30)
    current_budget = recent_data['spend'].sum()
    current_roas = recent_data['roas'].mean()
    
    scaling_recs = ai_assistant.get_scaling_recommendations(current_budget, current_roas)
    
    for rec in scaling_recs:
        st.markdown(f'<div class="ai-suggestion">', unsafe_allow_html=True)
        st.markdown(f"**Recommended Action:** {rec['action']}")
        st.write(f"**Budget Adjustment:** {rec['budget_increase']}")
        st.write(f"**Rationale:** {rec['rationale']}")
        st.write(f"**Monitoring:** {rec['monitoring']}")
        st.markdown('</div>', unsafe_allow_html=True)

def show_historical_analysis(historical_data):
    """Display historical performance analysis"""
    st.header("📊 Historical Performance Analysis")
    
    # Performance by day of week
    historical_data['day_of_week'] = historical_data['date'].dt.day_name()
    day_performance = historical_data.groupby('day_of_week').agg({
        'spend': 'mean',
        'revenue': 'mean',
        'roas': 'mean',
        'ctr': 'mean',
        'conversion_rate': 'mean'
    }).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(day_performance.reset_index(), x='day_of_week', y='roas',
                     title="Average ROAS by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(day_performance.reset_index(), x='day_of_week', y='spend',
                     title="Average Spend by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly trends
    historical_data['month'] = historical_data['date'].dt.month_name()
    monthly_performance = historical_data.groupby('month').agg({
        'spend': 'sum',
        'revenue': 'sum',
        'roas': 'mean'
    }).round(2)
    
    fig = px.line(monthly_performance.reset_index(), x='month', y='roas',
                  title="Monthly ROAS Trend")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("📈 Metric Correlations")
    correlation_matrix = historical_data[['spend', 'revenue', 'roas', 'ctr', 'cpc', 'conversion_rate']].corr()
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                    title="Correlation Matrix of Key Metrics")
    st.plotly_chart(fig, use_container_width=True)

def show_scenario_planning(projector, target_improvements):
    """Display scenario planning tools"""
    st.header("⚙️ Scenario Planning")
    
    st.write("Compare different scenarios to optimize your Meta Ads strategy:")
    
    # Create different scenarios
    scenarios = {
        "Current Performance": {"budget_change": 0, "improvements": {}},
        "Optimistic Growth": {"budget_change": 25, "improvements": {"ctr_improvement": 15, "conversion_rate_improvement": 20}},
        "Conservative Growth": {"budget_change": 10, "improvements": {"ctr_improvement": 5, "conversion_rate_improvement": 10}},
        "Cost Reduction Focus": {"budget_change": -10, "improvements": {"cpc_change": -15, "conversion_rate_improvement": 15}},
        "Aggressive Scaling": {"budget_change": 50, "improvements": {"ctr_improvement": 10, "conversion_rate_improvement": 15}}
    }
    
    scenario_results = {}
    
    for scenario_name, params in scenarios.items():
        projected_data = projector.calculate_projections(
            days_to_project=30,
            budget_change=params["budget_change"],
            target_improvements=params["improvements"]
        )
        
        scenario_results[scenario_name] = {
            "total_spend": projected_data['spend'].sum(),
            "total_revenue": projected_data['revenue'].sum(),
            "avg_roas": projected_data['roas'].mean(),
            "total_conversions": projected_data['conversions'].sum(),
            "avg_cpa": projected_data['cpa'].mean()
        }
    
    # Display scenario comparison
    scenario_df = pd.DataFrame(scenario_results).T
    scenario_df = scenario_df.round(2)
    
    st.subheader("📊 Scenario Comparison")
    st.dataframe(scenario_df, use_container_width=True)
    
    # Visualize scenario comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(scenario_df.reset_index(), x='index', y='avg_roas',
                     title="ROAS by Scenario", labels={'index': 'Scenario'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(scenario_df.reset_index(), x='index', y='total_revenue',
                     title="Revenue by Scenario", labels={'index': 'Scenario'})
        st.plotly_chart(fig, use_container_width=True)
    
    # ROI analysis
    st.subheader("💰 ROI Analysis")
    roi_data = []
    for scenario, data in scenario_results.items():
        profit = data['total_revenue'] - data['total_spend']
        roi_percent = (profit / data['total_spend']) * 100 if data['total_spend'] > 0 else 0
        roi_data.append({
            'Scenario': scenario,
            'Total Profit': profit,
            'ROI %': roi_percent
        })
    
    roi_df = pd.DataFrame(roi_data)
    fig = px.bar(roi_df, x='Scenario', y='ROI %', title="ROI Percentage by Scenario")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()