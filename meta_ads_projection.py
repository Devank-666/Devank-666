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

# Helper function to safely sum numeric columns in DataFrames
def safe_numeric_sum(df, exclude_columns=['date']):
    """
    Safely sum numeric columns in a DataFrame, excluding datetime and other non-numeric columns.
    
    Args:
        df (pd.DataFrame): DataFrame to sum
        exclude_columns (list): Columns to exclude from summation
    
    Returns:
        pd.Series: Sum of numeric columns
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove any explicitly excluded columns
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
    return df[numeric_columns].sum()

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
        self.historical_data = None
        self.available_metrics = []
        self.required_columns = ['date']
        
    def detect_and_validate_columns(self, df):
        """Dynamically detect available metrics in the CSV file"""
        available_columns = df.columns.tolist()
        
        # Essential column
        if 'date' not in available_columns:
            return False, ["date column is required"]
        
        # Detect available metrics
        metric_mapping = {
            'spend': ['spend', 'cost', 'amount_spent', 'ad_spend'],
            'impressions': ['impressions', 'impr', 'views'],
            'clicks': ['clicks', 'click', 'link_clicks'],
            'conversions': ['conversions', 'conv', 'purchases', 'results'],
            'revenue': ['revenue', 'purchase_value', 'conversion_value', 'sales'],
            'ctr': ['ctr', 'click_through_rate', 'clickthrough_rate'],
            'cpc': ['cpc', 'cost_per_click', 'avg_cpc'],
            'cpm': ['cpm', 'cost_per_mille', 'cost_per_1000_impressions'],
            'conversion_rate': ['conversion_rate', 'conv_rate', 'cvr'],
            'roas': ['roas', 'return_on_ad_spend', 'return_on_ads'],
            'cpa': ['cpa', 'cost_per_acquisition', 'cost_per_conversion'],
            'reach': ['reach', 'unique_reach'],
            'frequency': ['frequency', 'avg_frequency']
        }
        
        detected_metrics = {}
        for metric, possible_names in metric_mapping.items():
            for col_name in available_columns:
                if col_name.lower() in [name.lower() for name in possible_names]:
                    detected_metrics[metric] = col_name
                    break
        
        self.available_metrics = list(detected_metrics.keys())
        return True, detected_metrics
    
    def calculate_derived_metrics(self, df, detected_metrics):
        """Calculate missing metrics based on available data"""
        # Standardize column names
        for metric, col_name in detected_metrics.items():
            if col_name != metric:
                df[metric] = df[col_name]
        
        # Calculate derived metrics if base metrics are available
        if 'impressions' in detected_metrics and 'clicks' in detected_metrics and 'ctr' not in detected_metrics:
            df['ctr'] = (df['clicks'] / df['impressions'].replace(0, np.nan) * 100).round(2)
            self.available_metrics.append('ctr')
        
        if 'spend' in detected_metrics and 'clicks' in detected_metrics and 'cpc' not in detected_metrics:
            df['cpc'] = (df['spend'] / df['clicks'].replace(0, np.nan)).round(2)
            self.available_metrics.append('cpc')
        
        if 'spend' in detected_metrics and 'impressions' in detected_metrics and 'cpm' not in detected_metrics:
            df['cpm'] = (df['spend'] / df['impressions'].replace(0, np.nan) * 1000).round(2)
            self.available_metrics.append('cpm')
        
        if 'conversions' in detected_metrics and 'clicks' in detected_metrics and 'conversion_rate' not in detected_metrics:
            df['conversion_rate'] = (df['conversions'] / df['clicks'].replace(0, np.nan) * 100).round(2)
            self.available_metrics.append('conversion_rate')
        
        if 'revenue' in detected_metrics and 'spend' in detected_metrics and 'roas' not in detected_metrics:
            df['roas'] = (df['revenue'] / df['spend'].replace(0, np.nan)).round(2)
            self.available_metrics.append('roas')
        
        if 'spend' in detected_metrics and 'conversions' in detected_metrics and 'cpa' not in detected_metrics:
            df['cpa'] = (df['spend'] / df['conversions'].replace(0, np.nan)).round(2)
            self.available_metrics.append('cpa')
        
        # Clean infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
        
    def load_csv_data(self, uploaded_file):
        """Load and process Meta Ads data from uploaded CSV file"""
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Detect and validate columns
            is_valid, detected_metrics = self.detect_and_validate_columns(df)
            
            if not is_valid:
                st.error(f"Invalid CSV format: {detected_metrics}")
                st.info("Required: 'date' column. Recommended: spend, impressions, clicks, conversions, revenue")
                return False
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculate derived metrics
            df = self.calculate_derived_metrics(df, detected_metrics)
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Store the processed data
            self.historical_data = df
            
            st.success(f"✅ Successfully loaded {len(df)} rows of data!")
            st.info(f"📊 Detected metrics: {', '.join(self.available_metrics)}")
            st.info(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            
            return True
            
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            return False
    

        
    def calculate_projections(self, days_to_project=30, budget_change=0, target_improvements=None):
        """Calculate future projections based on available metrics in historical data"""
        if target_improvements is None:
            target_improvements = {}
            
        if self.historical_data is None or len(self.available_metrics) == 0:
            st.error("No data available for projections")
            return pd.DataFrame()
            
        # Prepare features for ML model
        data_copy = self.historical_data.copy()
        data_copy['day_of_week'] = data_copy['date'].dt.dayofweek
        data_copy['month'] = data_copy['date'].dt.month
        data_copy['day_of_year'] = data_copy['date'].dt.dayofyear
        
        # Features for prediction
        features = ['day_of_week', 'month', 'day_of_year']
        X = data_copy[features]
        
        # Determine which metrics to predict based on available data
        predictable_metrics = [metric for metric in self.available_metrics if metric in data_copy.columns and metric != 'date']
        
        if not predictable_metrics:
            st.error("No predictable metrics found in the data")
            return pd.DataFrame()
        
        # Train models for available metrics
        models = {}
        for metric in predictable_metrics:
            if data_copy[metric].notna().sum() > 5:  # Need at least 5 data points
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X, data_copy[metric])
                models[metric] = model
        
        # Generate future dates
        last_date = data_copy['date'].max()
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
        for metric in models.keys():
            predictions[metric] = models[metric].predict(X_future)
        
        # Apply improvements and budget changes
        budget_multiplier = 1 + (budget_change / 100)
        
        # Build projected data based on available metrics
        projected_data = []
        for i, date in enumerate(future_dates):
            row_data = {'date': date}
            
            # Apply predictions with improvements
            for metric in predictions.keys():
                base_value = predictions[metric][i]
                
                # Apply budget changes to volume metrics
                if metric in ['impressions', 'clicks', 'spend', 'conversions', 'revenue']:
                    base_value *= budget_multiplier
                
                # Apply specific improvements
                if metric == 'ctr':
                    base_value *= (1 + target_improvements.get('ctr_improvement', 0) / 100)
                elif metric == 'cpc':
                    base_value *= (1 + target_improvements.get('cpc_change', 0) / 100)
                elif metric == 'conversion_rate':
                    base_value *= (1 + target_improvements.get('conversion_rate_improvement', 0) / 100)
                elif metric == 'revenue':
                    base_value *= (1 + target_improvements.get('aov_improvement', 0) / 100)
                
                # Round appropriately
                if metric in ['impressions', 'clicks', 'conversions']:
                    row_data[metric] = max(0, int(base_value))
                else:
                    row_data[metric] = max(0, round(base_value, 2))
            
            # Calculate derived metrics if base metrics are available
            if 'impressions' in row_data and 'clicks' in row_data and 'ctr' not in row_data:
                row_data['ctr'] = round((row_data['clicks'] / row_data['impressions'] * 100) if row_data['impressions'] > 0 else 0, 2)
            
            if 'spend' in row_data and 'clicks' in row_data and 'cpc' not in row_data:
                row_data['cpc'] = round((row_data['spend'] / row_data['clicks']) if row_data['clicks'] > 0 else 0, 2)
            
            if 'conversions' in row_data and 'clicks' in row_data and 'conversion_rate' not in row_data:
                row_data['conversion_rate'] = round((row_data['conversions'] / row_data['clicks'] * 100) if row_data['clicks'] > 0 else 0, 2)
            
            if 'revenue' in row_data and 'spend' in row_data and 'roas' not in row_data:
                row_data['roas'] = round((row_data['revenue'] / row_data['spend']) if row_data['spend'] > 0 else 0, 2)
            
            if 'spend' in row_data and 'conversions' in row_data and 'cpa' not in row_data:
                row_data['cpa'] = round((row_data['spend'] / row_data['conversions']) if row_data['conversions'] > 0 else 0, 2)
            
            projected_data.append(row_data)
        
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
def initialize_ai_assistant():
    return AIAssistant()

ai_assistant = initialize_ai_assistant()

# Main App
def main():
    st.markdown('<h1 class="main-header">📊 Meta Ads Performance Projections & AI Assistant</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.header("📁 Upload Your Meta Ads Data")
    
    # File upload widget
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your Meta Ads data CSV file. Required: 'date' column. The system will automatically detect and work with available metrics."
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to start analyzing your Meta Ads data")
        st.markdown("### 📋 CSV Requirements:")
        st.markdown("""
        **Required Column:**
        - **date**: Date in YYYY-MM-DD format
        
        **Supported Metrics** (include any combination):
        - **spend/cost/amount_spent**: Ad spend amount
        - **impressions/impr/views**: Number of impressions  
        - **clicks/click/link_clicks**: Number of clicks
        - **conversions/conv/purchases**: Number of conversions
        - **revenue/purchase_value/sales**: Revenue generated
        - **ctr/click_through_rate**: Click-through rate
        - **cpc/cost_per_click**: Cost per click
        - **cpm/cost_per_mille**: Cost per thousand impressions
        - **conversion_rate/conv_rate**: Conversion rate
        - **roas/return_on_ad_spend**: Return on ad spend
        - **cpa/cost_per_acquisition**: Cost per acquisition
        - **reach/unique_reach**: Reach metrics
        - **frequency/avg_frequency**: Frequency metrics
        
        The system will automatically detect your metrics and calculate missing derived metrics where possible.
        """)
        return
    
    # Initialize projector and load data
    projector = MetaAdsProjector()
    
    if not projector.load_csv_data(uploaded_file):
        st.error("❌ Failed to load data. Please check your CSV file format.")
        return
    
    # Only proceed if we have valid data
    if projector.historical_data is None:
        st.warning("⚠️ No valid data available.")
        return
    
    # Display data summary
    st.subheader("📊 Data Summary")
    
    # Create dynamic columns based on available metrics
    summary_metrics = []
    if 'spend' in projector.available_metrics:
        summary_metrics.append(("Total Spend", f"${projector.historical_data['spend'].sum():,.2f}"))
    if 'revenue' in projector.available_metrics:
        summary_metrics.append(("Total Revenue", f"${projector.historical_data['revenue'].sum():,.2f}"))
    if 'impressions' in projector.available_metrics:
        summary_metrics.append(("Total Impressions", f"{projector.historical_data['impressions'].sum():,.0f}"))
    if 'clicks' in projector.available_metrics:
        summary_metrics.append(("Total Clicks", f"{projector.historical_data['clicks'].sum():,.0f}"))
    if 'conversions' in projector.available_metrics:
        summary_metrics.append(("Total Conversions", f"{projector.historical_data['conversions'].sum():,.0f}"))
    
    # Always show basic info
    basic_info = [
        ("Total Records", len(projector.historical_data)),
        ("Date Range", f"{(projector.historical_data['date'].max() - projector.historical_data['date'].min()).days} days")
    ]
    
    # Combine and display metrics
    all_metrics = basic_info + summary_metrics[:3]  # Limit to 5 total columns
    cols = st.columns(len(all_metrics))
    
    for i, (label, value) in enumerate(all_metrics):
        with cols[i]:
            st.metric(label, value)
    
    # Data preview
    with st.expander("📋 View Data Preview & Quality Indicators"):
        st.dataframe(projector.historical_data.head(10), use_container_width=True)
        
        # Show data quality indicators based on available metrics
        st.subheader("📈 Performance Indicators")
        
        quality_metrics = []
        if 'roas' in projector.available_metrics:
            avg_roas = projector.historical_data['roas'].mean()
            quality_metrics.append(("Average ROAS", f"{avg_roas:.2f}", "Good" if avg_roas >= 3.0 else "Needs Improvement"))
        
        if 'ctr' in projector.available_metrics:
            avg_ctr = projector.historical_data['ctr'].mean()
            quality_metrics.append(("Average CTR", f"{avg_ctr:.2f}%", "Good" if avg_ctr >= 1.5 else "Needs Improvement"))
        
        if 'conversion_rate' in projector.available_metrics:
            avg_conv_rate = projector.historical_data['conversion_rate'].mean()
            quality_metrics.append(("Average Conv. Rate", f"{avg_conv_rate:.2f}%", "Good" if avg_conv_rate >= 3.0 else "Needs Improvement"))
        
        if 'cpc' in projector.available_metrics:
            avg_cpc = projector.historical_data['cpc'].mean()
            quality_metrics.append(("Average CPC", f"${avg_cpc:.2f}", "Good" if avg_cpc <= 2.0 else "Needs Improvement"))
        
        if 'cpa' in projector.available_metrics:
            avg_cpa = projector.historical_data['cpa'].mean()
            quality_metrics.append(("Average CPA", f"${avg_cpa:.2f}", "Monitor"))
        
        if quality_metrics:
            qual_cols = st.columns(min(len(quality_metrics), 3))
            for i, (label, value, status) in enumerate(quality_metrics[:3]):
                with qual_cols[i]:
                    st.metric(label, value, delta=status)
        else:
            st.info("Upload data with performance metrics (ROAS, CTR, etc.) to see quality indicators.")
    
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
    # Use safe numeric sum to avoid datetime sum error
    projected_totals = safe_numeric_sum(projected_data)
    
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