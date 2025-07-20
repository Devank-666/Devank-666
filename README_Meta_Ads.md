# 📊 Meta Ads Performance Projections & AI Assistant

A comprehensive projection template for Meta Ads designed for performance marketing companies. This application provides advanced analytics, AI-driven insights, and performance optimization recommendations to maximize advertising ROI.

## 🚀 Features

### 📈 Performance Dashboard
- Real-time KPI tracking (ROAS, CPA, CTR, Conversion Rate)
- Revenue vs. Spend analysis
- Performance trend visualization
- 30-day rolling metrics

### 🔮 Advanced Projections
- Machine Learning-based forecasting using Random Forest algorithms
- Customizable projection periods (7-90 days)
- Budget scaling scenarios
- Performance improvement modeling

### 🤖 AI-Powered Insights
- Automated performance analysis
- Priority-based recommendations (High, Medium, Low)
- Actionable optimization strategies
- Budget scaling guidance based on current ROAS

### 📊 Historical Analysis
- Day-of-week performance patterns
- Monthly trend analysis
- Metric correlation matrices
- Seasonal performance insights

### ⚙️ Scenario Planning
- Multiple growth scenarios comparison
- ROI analysis across different strategies
- What-if analysis for budget and performance changes
- Risk assessment for scaling decisions

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project files**
   ```bash
   # Ensure you have the following files:
   # - meta_ads_projection.py
   # - requirements.txt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run meta_ads_projection.py
   ```

4. **Access the dashboard**
   - Open your browser and navigate to `http://localhost:8501`
   - Choose to upload your CSV file or use sample data

## 📋 How to Use

### 1. Data Input
- **Upload CSV File**: Upload your own Meta Ads data CSV file
- **Use Sample Data**: Use the built-in sample data for demonstration
- **Required CSV columns**: date, spend, impressions, clicks, conversions, revenue
- **Optional columns**: ctr, cpc, conversion_rate, roas, cpa (auto-calculated if missing)

### 2. Dashboard Overview
- View current performance metrics in the **📈 Dashboard** tab
- Monitor key KPIs: Spend, Revenue, ROAS, and CPA
- Analyze recent performance trends

### 3. Setting Up Projections
- Use the **sidebar controls** to configure projections:
  - **Days to Project**: 7-90 days
  - **Budget Change**: -50% to +100%
  - **Target Improvements**: CTR, Conversion Rate, CPC, AOV

### 4. Analyzing Projections
- Navigate to **🔮 Projections** tab
- Review projected revenue, spend, and ROAS
- Examine financial and volume metrics

### 5. AI Recommendations
- Check **🤖 AI Insights** tab for:
  - Priority-ranked performance issues
  - Specific action items for improvement
  - Budget scaling recommendations

### 6. Historical Analysis
- Use **📊 Historical Analysis** tab to:
  - Identify day-of-week patterns
  - Analyze monthly trends
  - Understand metric correlations

### 7. Scenario Planning
- Explore **⚙️ Scenario Planning** tab to:
  - Compare different growth strategies
  - Analyze ROI across scenarios
  - Make data-driven scaling decisions

## 📊 Key Metrics Explained

### ROAS (Return on Ad Spend)
- **Target**: 3.0+ (300% return)
- **Calculation**: Revenue ÷ Spend
- **Good**: 4.0+, **Acceptable**: 3.0-4.0, **Needs Improvement**: <3.0

### CTR (Click-Through Rate)
- **Target**: 1.5%+
- **Calculation**: (Clicks ÷ Impressions) × 100
- **Industry Average**: 1.0-2.0%

### CPA (Cost Per Acquisition)
- **Target**: Varies by industry
- **Calculation**: Spend ÷ Conversions
- **Lower is generally better**

### Conversion Rate
- **Target**: 3.0%+
- **Calculation**: (Conversions ÷ Clicks) × 100
- **Industry Average**: 2.0-4.0%

## 🎯 AI Recommendation Categories

### High Priority (🚨)
- ROAS below 3.0
- CTR below 1.0%
- Conversion Rate below 2.0%
- Immediate action required

### Medium Priority (⚠️)
- Moderate performance issues
- Optimization opportunities
- Scaling considerations

### Low Priority (✅)
- Performance maintenance
- Advanced optimization
- Scaling strategies

## 💡 Best Practices

### 1. Regular Monitoring
- Check dashboard daily
- Review AI recommendations weekly
- Analyze trends monthly

### 2. Data-Driven Decisions
- Use projections for budget planning
- Test recommended optimizations
- Monitor changes impact

### 3. Scaling Strategy
- Scale gradually when ROAS > 4.0
- Optimize before scaling if ROAS < 3.0
- Use scenario planning for major changes

### 4. Performance Optimization
- Focus on high-priority recommendations first
- Implement A/B tests for creative changes
- Monitor landing page performance

## 🔄 Customization Options

### Data Integration
Replace the sample data generation in `initialize_sample_data()` with your actual Meta Ads data:

```python
def load_your_data():
    # Replace with your data source
    return pd.read_csv('your_meta_ads_data.csv')
```

### AI Recommendations
Customize recommendation thresholds in the `AIAssistant` class:

```python
# Modify these values based on your industry standards
if avg_roas < 3.0:  # Change threshold here
    # Your custom recommendation logic
```

### Projection Models
Enhance the ML models in `calculate_projections()`:

```python
# Add more sophisticated models
from sklearn.ensemble import GradientBoostingRegressor
# Implement additional features and models
```

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade streamlit pandas plotly scikit-learn
   ```

2. **Data Loading Issues**
   - Ensure sample data generation is working
   - Check date ranges and data types

3. **Performance Issues**
   - Reduce projection days for faster processing
   - Check system memory for large datasets

4. **Visualization Problems**
   - Update plotly: `pip install --upgrade plotly`
   - Clear browser cache

## 📈 Advanced Features

### Custom Metrics
Add industry-specific metrics by modifying the data generation and analysis functions.

### API Integration
Connect to Meta Ads API for real-time data:
```python
# Example Meta Ads API integration
from facebook_business.api import FacebookAdsApi
# Implement API data fetching
```

### Export Functionality
Add data export capabilities:
```python
# CSV export
st.download_button(
    label="Download Projections",
    data=projected_data.to_csv(),
    file_name="meta_ads_projections.csv"
)
```

## 🤝 Support

For technical support or feature requests:
1. Check the troubleshooting section
2. Review the code comments for implementation details
3. Test with different parameter combinations

## 📄 License

This project is designed for performance marketing companies. Customize and adapt according to your specific needs and requirements.

---

**Note**: This application uses simulated data for demonstration. Replace with your actual Meta Ads data for production use.