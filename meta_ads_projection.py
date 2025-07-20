import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Meta Ads Budget Projector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .ai-recommendation {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

class SimpleBudgetProjector:
    def __init__(self):
        # Industry benchmarks
        self.benchmarks = {
            'avg_ctr': 1.5,  # Average CTR for Meta Ads
            'avg_conversion_rate': 3.2,  # Average conversion rate
            'avg_aov': 75,  # Average order value
            'avg_frequency': 2.1  # Average frequency
        }
    
    def calculate_projections(self, budget, cpc, days=30):
        """Calculate detailed projections based on budget and CPC"""
        
        # Daily budget
        daily_budget = budget / days
        
        # Basic calculations
        daily_clicks = daily_budget / cpc
        total_clicks = daily_clicks * days
        
        # Estimate impressions based on average CTR
        ctr = self.benchmarks['avg_ctr']
        daily_impressions = daily_clicks / (ctr / 100)
        total_impressions = daily_impressions * days
        
        # Estimate conversions
        conversion_rate = self.benchmarks['avg_conversion_rate']
        daily_conversions = daily_clicks * (conversion_rate / 100)
        total_conversions = daily_conversions * days
        
        # Estimate revenue
        aov = self.benchmarks['avg_aov']
        daily_revenue = daily_conversions * aov
        total_revenue = daily_revenue * days
        
        # Calculate derived metrics
        roas = total_revenue / budget if budget > 0 else 0
        cpa = budget / total_conversions if total_conversions > 0 else 0
        cpm = (budget / total_impressions) * 1000 if total_impressions > 0 else 0
        
        # Create daily breakdown
        dates = pd.date_range(start=datetime.now(), periods=days, freq='D')
        daily_data = []
        
        for i, date in enumerate(dates):
            # Add some realistic variance
            variance = np.random.normal(1, 0.1)
            daily_data.append({
                'date': date,
                'spend': round(daily_budget * variance, 2),
                'impressions': int(daily_impressions * variance),
                'clicks': int(daily_clicks * variance),
                'conversions': int(daily_conversions * variance),
                'revenue': round(daily_revenue * variance, 2),
                'ctr': round(ctr, 2),
                'cpc': round(cpc, 2),
                'conversion_rate': round(conversion_rate, 2),
                'roas': round(roas, 2),
                'cpa': round(cpa, 2),
                'cpm': round(cpm, 2)
            })
        
        projections_df = pd.DataFrame(daily_data)
        
        # Summary metrics
        summary = {
            'total_budget': budget,
            'total_clicks': int(total_clicks),
            'total_impressions': int(total_impressions),
            'total_conversions': int(total_conversions),
            'total_revenue': round(total_revenue, 2),
            'avg_ctr': round(ctr, 2),
            'avg_cpc': round(cpc, 2),
            'avg_conversion_rate': round(conversion_rate, 2),
            'roas': round(roas, 2),
            'cpa': round(cpa, 2),
            'cpm': round(cpm, 2),
            'profit': round(total_revenue - budget, 2),
            'profit_margin': round(((total_revenue - budget) / total_revenue * 100) if total_revenue > 0 else 0, 2)
        }
        
        return projections_df, summary
    
    def generate_ai_recommendations(self, summary, cpc, budget):
        """Generate AI recommendations based on projections"""
        recommendations = []
        
        # ROAS Analysis
        if summary['roas'] < 2.0:
            recommendations.append({
                'type': '🚨 Critical',
                'title': 'Low ROAS Alert',
                'message': f"Your projected ROAS of {summary['roas']:.2f} is below the 2.0 minimum threshold. Consider reducing CPC or improving conversion rates.",
                'action': 'Optimize targeting or reduce bid amounts'
            })
        elif summary['roas'] < 3.0:
            recommendations.append({
                'type': '⚠️ Warning',
                'title': 'ROAS Improvement Needed',
                'message': f"ROAS of {summary['roas']:.2f} is acceptable but could be better. Industry average is 3.0+.",
                'action': 'Test new ad creatives or landing pages'
            })
        else:
            recommendations.append({
                'type': '✅ Good',
                'title': 'Strong ROAS Performance',
                'message': f"Excellent ROAS of {summary['roas']:.2f}! Your campaign is performing well.",
                'action': 'Consider scaling budget to maximize profits'
            })
        
        # CPC Analysis
        if cpc > 2.5:
            recommendations.append({
                'type': '🚨 Critical',
                'title': 'High CPC Warning',
                'message': f"Your CPC of ${cpc:.2f} is quite high. This may impact profitability.",
                'action': 'Improve Quality Score, refine targeting, or test lower bids'
            })
        elif cpc > 1.5:
            recommendations.append({
                'type': '⚠️ Warning',
                'title': 'CPC Optimization Opportunity',
                'message': f"CPC of ${cpc:.2f} is moderate but could be optimized.",
                'action': 'Test different bidding strategies or audience refinements'
            })
        else:
            recommendations.append({
                'type': '✅ Good',
                'title': 'Competitive CPC',
                'message': f"Your CPC of ${cpc:.2f} is competitive and cost-effective.",
                'action': 'Maintain current strategy and monitor performance'
            })
        
        # Conversion Analysis
        if summary['avg_conversion_rate'] < 2.0:
            recommendations.append({
                'type': '🚨 Critical',
                'title': 'Low Conversion Rate',
                'message': f"Conversion rate of {summary['avg_conversion_rate']:.2f}% needs improvement.",
                'action': 'Optimize landing pages, improve product pages, or refine targeting'
            })
        elif summary['avg_conversion_rate'] < 3.0:
            recommendations.append({
                'type': '⚠️ Warning',
                'title': 'Conversion Rate Below Average',
                'message': f"Conversion rate of {summary['avg_conversion_rate']:.2f}% is below industry average of 3.2%.",
                'action': 'A/B test landing pages or checkout process'
            })
        
        # Budget Recommendations
        if budget < 1000:
            recommendations.append({
                'type': '💡 Suggestion',
                'title': 'Budget Scale Opportunity',
                'message': f"With a budget of ${budget:,.2f}, consider scaling if ROAS is strong.",
                'action': 'Gradually increase daily budget by 20-30% if performance is good'
            })
        
        # Profit Analysis
        if summary['profit'] < 0:
            recommendations.append({
                'type': '🚨 Critical',
                'title': 'Negative Profit Projection',
                'message': f"Projected loss of ${abs(summary['profit']):,.2f}. Immediate optimization needed.",
                'action': 'Reduce budget, lower CPC, or improve conversion funnel'
            })
        
        return recommendations

def main():
    st.markdown('<h1 class="main-header">📊 Meta Ads Budget Projector</h1>', unsafe_allow_html=True)
    
    # Input Section
    st.header("💰 Budget & CPC Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        budget = st.number_input(
            "Total Budget ($)",
            min_value=100.0,
            max_value=100000.0,
            value=5000.0,
            step=100.0,
            help="Enter your total advertising budget"
        )
    
    with col2:
        cpc = st.number_input(
            "Target CPC ($)",
            min_value=0.10,
            max_value=10.0,
            value=1.25,
            step=0.05,
            help="Enter your target cost per click"
        )
    
    with col3:
        days = st.number_input(
            "Projection Period (Days)",
            min_value=7,
            max_value=90,
            value=30,
            step=1,
            help="Number of days to project"
        )
    
    if st.button("🚀 Generate Projection", type="primary"):
        projector = SimpleBudgetProjector()
        projections_df, summary = projector.calculate_projections(budget, cpc, days)
        
        # Display Summary Metrics
        st.header("📊 Projection Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Clicks", f"{summary['total_clicks']:,}")
            st.metric("Total Impressions", f"{summary['total_impressions']:,}")
        
        with col2:
            st.metric("Total Conversions", f"{summary['total_conversions']:,}")
            st.metric("Conversion Rate", f"{summary['avg_conversion_rate']}%")
        
        with col3:
            st.metric("Total Revenue", f"${summary['total_revenue']:,.2f}")
            st.metric("ROAS", f"{summary['roas']:.2f}x")
        
        with col4:
            st.metric("Profit/Loss", f"${summary['profit']:,.2f}")
            st.metric("CPA", f"${summary['cpa']:.2f}")
        
        # Performance Charts
        st.header("📈 Performance Visualization")
        
        tab1, tab2, tab3 = st.tabs(["💰 Spend & Revenue", "👆 Clicks & Conversions", "📊 Key Metrics"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=projections_df['date'], y=projections_df['spend'], 
                                   mode='lines+markers', name='Daily Spend', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=projections_df['date'], y=projections_df['revenue'], 
                                   mode='lines+markers', name='Daily Revenue', line=dict(color='green')))
            fig.update_layout(title='Daily Spend vs Revenue', xaxis_title='Date', yaxis_title='Amount ($)')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=projections_df['date'], y=projections_df['clicks'], 
                                   mode='lines+markers', name='Daily Clicks', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=projections_df['date'], y=projections_df['conversions'], 
                                   mode='lines+markers', name='Daily Conversions', line=dict(color='orange')))
            fig.update_layout(title='Daily Clicks vs Conversions', xaxis_title='Date', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # ROAS Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = summary['roas'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "ROAS"},
                    delta = {'reference': 3.0},
                    gauge = {'axis': {'range': [None, 6]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 2], 'color': "lightgray"},
                               {'range': [2, 3], 'color': "yellow"},
                               {'range': [3, 6], 'color': "green"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 2.0}}))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Conversion Rate Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = summary['avg_conversion_rate'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Conversion Rate (%)"},
                    delta = {'reference': 3.2},
                    gauge = {'axis': {'range': [None, 8]},
                           'bar': {'color': "darkgreen"},
                           'steps': [
                               {'range': [0, 2], 'color': "lightgray"},
                               {'range': [2, 3.2], 'color': "yellow"},
                               {'range': [3.2, 8], 'color': "green"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 2.0}}))
                st.plotly_chart(fig, use_container_width=True)
        
        # AI Recommendations
        st.header("🤖 AI Recommendations")
        
        recommendations = projector.generate_ai_recommendations(summary, cpc, budget)
        
        for rec in recommendations:
            if rec['type'] == '🚨 Critical':
                st.error(f"**{rec['title']}**\n\n{rec['message']}\n\n**Action:** {rec['action']}")
            elif rec['type'] == '⚠️ Warning':
                st.warning(f"**{rec['title']}**\n\n{rec['message']}\n\n**Action:** {rec['action']}")
            elif rec['type'] == '✅ Good':
                st.success(f"**{rec['title']}**\n\n{rec['message']}\n\n**Action:** {rec['action']}")
            else:
                st.info(f"**{rec['title']}**\n\n{rec['message']}\n\n**Action:** {rec['action']}")
        
        # Daily Breakdown Table
        st.header("📅 Daily Breakdown")
        st.dataframe(projections_df, use_container_width=True)

if __name__ == "__main__":
    main()