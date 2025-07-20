"""
Meta Ads Projection Template - Demo Examples
This script demonstrates various features of the Meta Ads projection system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_sample_campaigns():
    """Generate sample campaign data for demonstration"""
    campaigns = [
        {
            'campaign_name': 'Holiday Sale Campaign',
            'budget': 10000,
            'target_roas': 4.5,
            'industry': 'E-commerce',
            'audience_size': 500000,
            'optimization_goal': 'Purchase'
        },
        {
            'campaign_name': 'Brand Awareness Campaign',
            'budget': 5000,
            'target_roas': 2.5,
            'industry': 'Fashion',
            'audience_size': 1000000,
            'optimization_goal': 'Brand Awareness'
        },
        {
            'campaign_name': 'Lead Generation Campaign',
            'budget': 8000,
            'target_roas': 6.0,
            'industry': 'SaaS',
            'audience_size': 250000,
            'optimization_goal': 'Lead Generation'
        }
    ]
    return campaigns

def calculate_performance_projections(campaign_data, projection_days=30):
    """Calculate performance projections for a given campaign"""
    
    daily_budget = campaign_data['budget'] / projection_days
    
    # Industry benchmarks (simplified)
    benchmarks = {
        'E-commerce': {'ctr': 1.8, 'cpc': 1.2, 'conversion_rate': 3.5},
        'Fashion': {'ctr': 1.5, 'cpc': 1.4, 'conversion_rate': 2.8},
        'SaaS': {'ctr': 2.2, 'cpc': 2.1, 'conversion_rate': 4.2}
    }
    
    industry = campaign_data['industry']
    benchmark = benchmarks.get(industry, benchmarks['E-commerce'])
    
    # Calculate daily metrics
    daily_impressions = daily_budget / benchmark['cpc'] / (benchmark['ctr'] / 100) * 100
    daily_clicks = daily_impressions * (benchmark['ctr'] / 100)
    daily_spend = daily_clicks * benchmark['cpc']
    daily_conversions = daily_clicks * (benchmark['conversion_rate'] / 100)
    
    # Estimate revenue based on target ROAS
    daily_revenue = daily_spend * campaign_data['target_roas']
    
    projection = {
        'campaign_name': campaign_data['campaign_name'],
        'daily_metrics': {
            'impressions': int(daily_impressions),
            'clicks': int(daily_clicks),
            'spend': round(daily_spend, 2),
            'conversions': int(daily_conversions),
            'revenue': round(daily_revenue, 2),
            'ctr': benchmark['ctr'],
            'cpc': benchmark['cpc'],
            'conversion_rate': benchmark['conversion_rate'],
            'roas': campaign_data['target_roas']
        },
        'monthly_totals': {
            'total_impressions': int(daily_impressions * projection_days),
            'total_clicks': int(daily_clicks * projection_days),
            'total_spend': round(daily_spend * projection_days, 2),
            'total_conversions': int(daily_conversions * projection_days),
            'total_revenue': round(daily_revenue * projection_days, 2)
        }
    }
    
    return projection

def ai_optimization_suggestions(performance_data):
    """Generate AI-driven optimization suggestions"""
    suggestions = []
    
    current_roas = performance_data['daily_metrics']['roas']
    current_ctr = performance_data['daily_metrics']['ctr']
    current_conversion_rate = performance_data['daily_metrics']['conversion_rate']
    
    # ROAS Analysis
    if current_roas < 3.0:
        suggestions.append({
            'priority': 'HIGH',
            'category': 'ROAS Optimization',
            'issue': f'ROAS of {current_roas} is below the 3:1 minimum threshold',
            'recommendations': [
                'Review and pause poor-performing ad sets',
                'Optimize landing pages for better conversion rates',
                'Implement retargeting campaigns for warm audiences',
                'Test higher-value product promotions'
            ]
        })
    elif current_roas < 4.0:
        suggestions.append({
            'priority': 'MEDIUM',
            'category': 'ROAS Scaling',
            'issue': f'ROAS of {current_roas} has room for improvement',
            'recommendations': [
                'Increase budgets on top-performing ad sets',
                'Expand successful audiences with lookalikes',
                'Test new creative variations',
                'Implement dynamic product ads'
            ]
        })
    
    # CTR Analysis
    if current_ctr < 1.0:
        suggestions.append({
            'priority': 'HIGH',
            'category': 'Creative Optimization',
            'issue': f'CTR of {current_ctr}% is significantly below industry average',
            'recommendations': [
                'Refresh ad creative with more engaging visuals',
                'Test video ads vs static images',
                'Improve ad copy with stronger calls-to-action',
                'Review audience targeting for relevance'
            ]
        })
    elif current_ctr < 1.5:
        suggestions.append({
            'priority': 'MEDIUM',
            'category': 'Creative Testing',
            'issue': f'CTR of {current_ctr}% can be improved',
            'recommendations': [
                'A/B test new creative concepts',
                'Test different ad formats (carousel, video, etc.)',
                'Optimize ad placements'
            ]
        })
    
    # Conversion Rate Analysis
    if current_conversion_rate < 2.0:
        suggestions.append({
            'priority': 'HIGH',
            'category': 'Landing Page Optimization',
            'issue': f'Conversion rate of {current_conversion_rate}% needs immediate attention',
            'recommendations': [
                'Conduct landing page speed optimization',
                'Simplify checkout process',
                'Add trust signals and social proof',
                'Implement exit-intent popups'
            ]
        })
    elif current_conversion_rate < 3.0:
        suggestions.append({
            'priority': 'MEDIUM',
            'category': 'User Experience',
            'issue': f'Conversion rate of {current_conversion_rate}% has improvement potential',
            'recommendations': [
                'Test different landing page layouts',
                'Optimize mobile experience',
                'Add customer testimonials',
                'Implement live chat support'
            ]
        })
    
    return suggestions

def run_demo():
    """Run a complete demonstration of the Meta Ads projection system"""
    
    print("=" * 60)
    print("🚀 META ADS PROJECTION TEMPLATE DEMO")
    print("=" * 60)
    print()
    
    # Generate sample campaigns
    campaigns = generate_sample_campaigns()
    
    print("📊 CAMPAIGN ANALYSIS & PROJECTIONS")
    print("-" * 40)
    print()
    
    all_projections = []
    
    for i, campaign in enumerate(campaigns, 1):
        print(f"Campaign {i}: {campaign['campaign_name']}")
        print(f"Budget: ${campaign['budget']:,}")
        print(f"Target ROAS: {campaign['target_roas']}")
        print(f"Industry: {campaign['industry']}")
        print()
        
        # Calculate projections
        projection = calculate_performance_projections(campaign)
        all_projections.append(projection)
        
        # Display daily metrics
        daily = projection['daily_metrics']
        print("Daily Projected Metrics:")
        print(f"  • Impressions: {daily['impressions']:,}")
        print(f"  • Clicks: {daily['clicks']:,}")
        print(f"  • Spend: ${daily['spend']:,.2f}")
        print(f"  • Conversions: {daily['conversions']:,}")
        print(f"  • Revenue: ${daily['revenue']:,.2f}")
        print(f"  • CTR: {daily['ctr']:.2f}%")
        print(f"  • CPC: ${daily['cpc']:.2f}")
        print(f"  • Conversion Rate: {daily['conversion_rate']:.2f}%")
        print(f"  • ROAS: {daily['roas']:.2f}")
        print()
        
        # Display monthly totals
        monthly = projection['monthly_totals']
        print("30-Day Projected Totals:")
        print(f"  • Total Spend: ${monthly['total_spend']:,.2f}")
        print(f"  • Total Revenue: ${monthly['total_revenue']:,.2f}")
        print(f"  • Total Profit: ${monthly['total_revenue'] - monthly['total_spend']:,.2f}")
        print(f"  • Total Conversions: {monthly['total_conversions']:,}")
        print()
        
        # Generate AI suggestions
        suggestions = ai_optimization_suggestions(projection)
        
        if suggestions:
            print("🤖 AI OPTIMIZATION SUGGESTIONS:")
            for suggestion in suggestions:
                priority_emoji = "🚨" if suggestion['priority'] == 'HIGH' else "⚠️"
                print(f"{priority_emoji} {suggestion['priority']} PRIORITY - {suggestion['category']}")
                print(f"   Issue: {suggestion['issue']}")
                print("   Recommendations:")
                for rec in suggestion['recommendations']:
                    print(f"     • {rec}")
                print()
        else:
            print("✅ Campaign performance looks good! Focus on scaling strategies.")
            print()
        
        print("=" * 60)
        print()
    
    # Summary Analysis
    total_spend = sum(p['monthly_totals']['total_spend'] for p in all_projections)
    total_revenue = sum(p['monthly_totals']['total_revenue'] for p in all_projections)
    total_profit = total_revenue - total_spend
    overall_roas = total_revenue / total_spend if total_spend > 0 else 0
    
    print("📈 PORTFOLIO SUMMARY")
    print("-" * 40)
    print(f"Total Monthly Spend: ${total_spend:,.2f}")
    print(f"Total Monthly Revenue: ${total_revenue:,.2f}")
    print(f"Total Monthly Profit: ${total_profit:,.2f}")
    print(f"Overall Portfolio ROAS: {overall_roas:.2f}")
    print()
    
    # Portfolio recommendations
    if overall_roas >= 4.0:
        print("🟢 EXCELLENT: Portfolio performance allows for aggressive scaling")
        print("   • Increase budgets by 25-50%")
        print("   • Expand to new audiences and platforms")
        print("   • Test new campaign objectives")
    elif overall_roas >= 3.0:
        print("🟡 GOOD: Portfolio performance allows for conservative scaling")
        print("   • Increase budgets by 10-20%")
        print("   • Focus on optimizing underperforming campaigns")
        print("   • Test new creative formats")
    else:
        print("🔴 NEEDS ATTENTION: Focus on optimization before scaling")
        print("   • Pause poor-performing campaigns")
        print("   • Implement immediate optimization strategies")
        print("   • Review audience targeting and creative")
    
    print()
    print("=" * 60)
    print("✅ Demo completed! Run the full Streamlit app for interactive analysis.")
    print("   Command: ./run_meta_ads_app.sh")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()