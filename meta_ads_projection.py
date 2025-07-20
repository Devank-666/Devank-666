import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import altair as alt
import json
import random

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
    
    def get_performance_context(self, summary, cpc, budget, projections_df):
        """Get context about the current projections for AI assistant"""
        context = f"""
        Current Meta Ads Projection Analysis:
        
        BUDGET & SETUP:
        - Total Budget: ${budget:,.2f}
        - Target CPC: ${cpc:.2f}
        - Projection Period: {len(projections_df)} days
        
        PROJECTED PERFORMANCE:
        - Total Clicks: {summary['total_clicks']:,}
        - Total Impressions: {summary['total_impressions']:,}
        - Total Conversions: {summary['total_conversions']:,}
        - Total Revenue: ${summary['total_revenue']:,.2f}
        - Profit/Loss: ${summary['profit']:,.2f}
        
        KEY METRICS:
        - ROAS: {summary['roas']:.2f}x
        - CPA: ${summary['cpa']:.2f}
        - CTR: {summary['avg_ctr']:.2f}%
        - Conversion Rate: {summary['avg_conversion_rate']:.2f}%
        - CPM: ${summary['cpm']:.2f}
        
        PERFORMANCE BENCHMARKS:
        - Industry Average ROAS: 3.0x
        - Industry Average CTR: 1.5%
        - Industry Average Conversion Rate: 3.2%
        - Recommended CPC Range: $0.50 - $2.00
        
        DAILY PERFORMANCE:
        - Average Daily Spend: ${budget/len(projections_df):.2f}
        - Average Daily Revenue: ${summary['total_revenue']/len(projections_df):.2f}
        - Average Daily Clicks: {summary['total_clicks']//len(projections_df):,}
        - Average Daily Conversions: {summary['total_conversions']//len(projections_df):,}
        """
        return context

class AIAssistant:
    def __init__(self):
        self.context = ""
        self.conversation_history = []
        
    def set_context(self, context):
        """Set the current projection context for the AI assistant"""
        self.context = context
    
    def generate_response(self, user_question):
        """Generate AI response based on user question and projection context"""
        
        # Convert question to lowercase for pattern matching
        question_lower = user_question.lower()
        
        # Initialize response
        response = ""
        
        # Parse the context to extract key metrics
        lines = self.context.strip().split('\n')
        metrics = {}
        for line in lines:
            if ':' in line and any(char.isdigit() for char in line):
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip().replace('-', '').strip()
                    value = parts[1].strip()
                    metrics[key.lower()] = value
        
        # Question categories and responses
        if any(word in question_lower for word in ['roas', 'return', 'revenue ratio']):
            roas_value = self.extract_number(metrics.get('roas', '0'))
            if roas_value < 2.0:
                response = f"🚨 **ROAS Analysis**: Your projected ROAS of {roas_value:.2f}x is concerning. Here's what you need to know:\n\n" \
                          f"• **Current Status**: Below the 2.0x minimum threshold\n" \
                          f"• **Industry Benchmark**: 3.0x is considered good\n" \
                          f"• **Immediate Actions**: Reduce CPC, improve landing pages, or refine targeting\n" \
                          f"• **Risk**: You may lose money at this ROAS level\n" \
                          f"• **Priority**: HIGH - Address immediately before campaign launch"
            elif roas_value < 3.0:
                response = f"⚠️ **ROAS Analysis**: Your projected ROAS of {roas_value:.2f}x is acceptable but needs improvement:\n\n" \
                          f"• **Current Status**: Above break-even but below optimal\n" \
                          f"• **Industry Benchmark**: You're {3.0-roas_value:.2f}x below the 3.0x industry average\n" \
                          f"• **Optimization**: Test new ad creatives, improve landing pages\n" \
                          f"• **Potential**: Good foundation to build upon\n" \
                          f"• **Next Steps**: Focus on conversion rate optimization"
            else:
                response = f"✅ **ROAS Analysis**: Excellent ROAS of {roas_value:.2f}x! You're performing well:\n\n" \
                          f"• **Current Status**: {roas_value-3.0:.2f}x above industry benchmark\n" \
                          f"• **Performance**: Strong profitability indicators\n" \
                          f"• **Opportunity**: Consider scaling budget to maximize profits\n" \
                          f"• **Maintenance**: Monitor performance while scaling\n" \
                          f"• **Growth Strategy**: Expand to similar audiences"
        
        elif any(word in question_lower for word in ['cpc', 'cost per click', 'click cost']):
            cpc_value = self.extract_number(metrics.get('target cpc', '0'))
            response = f"💰 **CPC Analysis**: Your target CPC of ${cpc_value:.2f} analysis:\n\n"
            if cpc_value > 2.5:
                response += f"🚨 **Status**: High CPC - may impact profitability\n" \
                           f"• **Recommendation**: Reduce bids or improve Quality Score\n" \
                           f"• **Actions**: Refine targeting, improve ad relevance\n" \
                           f"• **Risk**: High acquisition costs\n" \
                           f"• **Target**: Aim for $1.50-$2.00 range"
            elif cpc_value > 1.5:
                response += f"⚠️ **Status**: Moderate CPC - room for optimization\n" \
                           f"• **Opportunity**: Test lower bid strategies\n" \
                           f"• **Actions**: Audience refinement, A/B test ads\n" \
                           f"• **Potential**: 15-25% cost reduction possible\n" \
                           f"• **Monitor**: Quality Score and relevance metrics"
            else:
                response += f"✅ **Status**: Competitive and cost-effective\n" \
                           f"• **Performance**: Well within optimal range\n" \
                           f"• **Strategy**: Maintain current approach\n" \
                           f"• **Opportunity**: Consider scaling volume\n" \
                           f"• **Advantage**: Cost efficiency for growth"
        
        elif any(word in question_lower for word in ['conversion', 'convert', 'cvr']):
            conv_rate = self.extract_number(metrics.get('conversion rate', '0'))
            response = f"🎯 **Conversion Rate Analysis**: Your projected {conv_rate:.2f}% conversion rate:\n\n"
            if conv_rate < 2.0:
                response += f"🚨 **Status**: Below industry standards\n" \
                           f"• **Benchmark Gap**: Industry average is 3.2%\n" \
                           f"• **Impact**: Significantly limiting campaign profitability\n" \
                           f"• **Priority Actions**: Landing page optimization, UX improvements\n" \
                           f"• **Testing**: A/B test checkout process, forms, CTAs\n" \
                           f"• **Quick Wins**: Add trust signals, social proof, testimonials"
            elif conv_rate < 3.2:
                response += f"⚠️ **Status**: Below industry average\n" \
                           f"• **Gap**: {3.2-conv_rate:.1f}% below benchmark\n" \
                           f"• **Opportunity**: Significant improvement potential\n" \
                           f"• **Focus**: Landing page and funnel optimization\n" \
                           f"• **Impact**: Could improve ROAS by {((3.2/conv_rate)-1)*100:.0f}%"
            else:
                response += f"✅ **Status**: Meeting or exceeding benchmarks\n" \
                           f"• **Performance**: {conv_rate-3.2:.1f}% above industry average\n" \
                           f"• **Strength**: Strong funnel optimization\n" \
                           f"• **Strategy**: Focus on scaling successful elements\n" \
                           f"• **Growth**: Test premium offerings or upsells"
        
        elif any(word in question_lower for word in ['budget', 'spend', 'money', 'cost']):
            total_budget = self.extract_number(metrics.get('total budget', '0'))
            profit = self.extract_number(metrics.get('profit/loss', '0'))
            response = f"💰 **Budget Analysis**: Your ${total_budget:,.0f} budget breakdown:\n\n"
            if profit < 0:
                response += f"🚨 **Profit Projection**: ${abs(profit):,.0f} loss projected\n" \
                           f"• **Risk Level**: HIGH - Campaign may lose money\n" \
                           f"• **Immediate Action**: Reduce budget or optimize performance\n" \
                           f"• **Options**: Lower CPC, improve conversion rate, or pause\n" \
                           f"• **Break-even**: Need {abs(profit/total_budget)*100:.0f}% improvement in ROAS"
            else:
                response += f"✅ **Profit Projection**: ${profit:,.0f} profit expected\n" \
                           f"• **ROI**: {(profit/total_budget)*100:.1f}% return on investment\n" \
                           f"• **Performance**: Budget is well-allocated\n" \
                           f"• **Scaling**: Consider increasing budget if ROAS remains stable\n" \
                           f"• **Growth**: {20}% budget increase could add ${profit*0.2:,.0f} profit"
        
        elif any(word in question_lower for word in ['optimize', 'improve', 'better', 'increase']):
            roas_value = self.extract_number(metrics.get('roas', '0'))
            conv_rate = self.extract_number(metrics.get('conversion rate', '0'))
            cpc_value = self.extract_number(metrics.get('target cpc', '0'))
            
            response = f"🚀 **Optimization Recommendations**: Top priorities for improvement:\n\n"
            
            recommendations = []
            if roas_value < 3.0:
                recommendations.append("**1. ROAS Improvement** - Focus on conversion rate and AOV optimization")
            if conv_rate < 3.2:
                recommendations.append("**2. Conversion Rate** - Landing page and funnel optimization")
            if cpc_value > 2.0:
                recommendations.append("**3. CPC Reduction** - Improve Quality Score and targeting")
            
            if not recommendations:
                recommendations.append("**1. Scale Performance** - Increase budget gradually")
                recommendations.append("**2. Audience Expansion** - Test lookalike audiences")
                recommendations.append("**3. Creative Testing** - A/B test new ad formats")
            
            for rec in recommendations[:3]:
                response += f"• {rec}\n"
            
            response += f"\n**Quick Wins**:\n" \
                       f"• Add urgency/scarcity to landing pages\n" \
                       f"• Test mobile-first ad creatives\n" \
                       f"• Implement retargeting campaigns\n" \
                       f"• A/B test headlines and CTAs"
        
        elif any(word in question_lower for word in ['scale', 'scaling', 'increase budget', 'grow']):
            roas_value = self.extract_number(metrics.get('roas', '0'))
            total_budget = self.extract_number(metrics.get('total budget', '0'))
            
            response = f"📈 **Scaling Strategy**: Budget scaling recommendations:\n\n"
            
            if roas_value >= 3.0:
                response += f"✅ **Ready to Scale**: ROAS of {roas_value:.2f}x supports growth\n" \
                           f"• **Scaling Approach**: Increase budget by 20-30% weekly\n" \
                           f"• **Monitor**: ROAS should stay above 2.5x while scaling\n" \
                           f"• **Target**: Scale to ${total_budget*1.5:,.0f} if performance holds\n" \
                           f"• **Risk Management**: Scale back if ROAS drops below 2.8x"
            elif roas_value >= 2.0:
                response += f"⚠️ **Conservative Scaling**: ROAS of {roas_value:.2f}x allows limited growth\n" \
                           f"• **Approach**: 10-15% weekly increases maximum\n" \
                           f"• **Priority**: Optimize performance before aggressive scaling\n" \
                           f"• **Safety**: Keep close eye on profitability metrics\n" \
                           f"• **Goal**: Improve ROAS to 3.0x before major scaling"
            else:
                response += f"🚨 **Not Ready to Scale**: ROAS of {roas_value:.2f}x too low\n" \
                           f"• **Priority**: Optimize performance first\n" \
                           f"• **Action**: Focus on conversion rate and CPC improvements\n" \
                           f"• **Target**: Achieve 2.5x+ ROAS before scaling\n" \
                           f"• **Risk**: Scaling now would increase losses"
        
        elif any(word in question_lower for word in ['click', 'impression', 'traffic']):
            clicks = metrics.get('total clicks', '0').replace(',', '')
            impressions = metrics.get('total impressions', '0').replace(',', '')
            ctr = self.extract_number(metrics.get('ctr', '0'))
            
            response = f"👆 **Traffic Analysis**: Your projected traffic metrics:\n\n" \
                      f"• **Total Clicks**: {clicks} over projection period\n" \
                      f"• **Total Impressions**: {impressions}\n" \
                      f"• **CTR**: {ctr:.2f}% (Industry benchmark: 1.5%)\n\n"
            
            if ctr < 1.0:
                response += f"🚨 **CTR Issue**: Significantly below benchmark\n" \
                           f"• **Impact**: Poor ad relevance and higher costs\n" \
                           f"• **Actions**: Refresh creative, improve targeting\n" \
                           f"• **Priority**: HIGH - affects all other metrics"
            elif ctr < 1.5:
                response += f"⚠️ **CTR Opportunity**: Below industry average\n" \
                           f"• **Potential**: {((1.5/ctr)-1)*100:.0f}% improvement possible\n" \
                           f"• **Actions**: A/B test headlines and visuals\n" \
                           f"• **Impact**: Better CTR = lower CPC"
            else:
                response += f"✅ **CTR Performance**: Above benchmark\n" \
                           f"• **Strength**: Strong ad relevance\n" \
                           f"• **Advantage**: Lower costs and better reach\n" \
                           f"• **Strategy**: Scale successful creatives"
        
        else:
            # General analysis if no specific category matches
            roas_value = self.extract_number(metrics.get('roas', '0'))
            profit = self.extract_number(metrics.get('profit/loss', '0'))
            
            response = f"📊 **Campaign Analysis**: Based on your projections:\n\n" \
                      f"**Performance Summary**:\n" \
                      f"• ROAS: {roas_value:.2f}x ({'✅ Good' if roas_value >= 3.0 else '⚠️ Needs improvement' if roas_value >= 2.0 else '🚨 Critical'})\n" \
                      f"• Profit: ${profit:,.0f} ({'✅ Profitable' if profit > 0 else '🚨 Loss projected'})\n\n" \
                      f"**Key Insights**:\n"
            
            if profit > 0 and roas_value >= 3.0:
                response += f"• Strong campaign setup with good profitability\n" \
                           f"• Consider scaling budget gradually\n" \
                           f"• Monitor performance closely while growing"
            elif profit > 0:
                response += f"• Profitable but room for optimization\n" \
                           f"• Focus on improving ROAS before scaling\n" \
                           f"• Test conversion rate improvements"
            else:
                response += f"• Campaign needs optimization before launch\n" \
                           f"• Reduce CPC or improve conversion rate\n" \
                           f"• Consider lowering initial budget"
        
        # Add the question and response to conversation history
        self.conversation_history.append({
            'question': user_question,
            'response': response,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        return response
    
    def extract_number(self, text):
        """Extract numeric value from text string"""
        try:
            # Remove common characters and extract number
            import re
            numbers = re.findall(r'[\d.]+', str(text))
            if numbers:
                return float(numbers[0])
            return 0.0
        except:
            return 0.0

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
        
        # AI Assistant
        st.header("🤖 AI Assistant - Ask Any Questions!")
        
        # Initialize AI assistant and set context
        if 'ai_assistant' not in st.session_state:
            st.session_state.ai_assistant = AIAssistant()
        
        # Set context for the AI assistant
        context = projector.get_performance_context(summary, cpc, budget, projections_df)
        st.session_state.ai_assistant.set_context(context)
        
        # Quick question buttons
        st.subheader("💡 Quick Questions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📈 How's my ROAS?"):
                response = st.session_state.ai_assistant.generate_response("How is my ROAS performance?")
                st.markdown(f"**AI Response:**\n\n{response}")
        
        with col2:
            if st.button("💰 Budget analysis?"):
                response = st.session_state.ai_assistant.generate_response("Can you analyze my budget and profit projections?")
                st.markdown(f"**AI Response:**\n\n{response}")
        
        with col3:
            if st.button("🚀 How to optimize?"):
                response = st.session_state.ai_assistant.generate_response("How can I optimize my campaign performance?")
                st.markdown(f"**AI Response:**\n\n{response}")
        
        with col4:
            if st.button("📊 Should I scale?"):
                response = st.session_state.ai_assistant.generate_response("Should I scale my budget and how?")
                st.markdown(f"**AI Response:**\n\n{response}")
        
        # Custom question input
        st.subheader("❓ Ask Your Own Question")
        user_question = st.text_area(
            "Type your question about the campaign projections:",
            placeholder="E.g., What's causing my low conversion rate? How can I improve my CPC? Is this budget sufficient for my goals?",
            height=100
        )
        
        if st.button("🎯 Get AI Answer", type="primary") and user_question.strip():
            with st.spinner("🤖 AI is analyzing your question..."):
                response = st.session_state.ai_assistant.generate_response(user_question)
                st.markdown("### 🤖 AI Response:")
                st.markdown(response)
        
        # Conversation History
        if hasattr(st.session_state.ai_assistant, 'conversation_history') and st.session_state.ai_assistant.conversation_history:
            with st.expander("💬 Conversation History"):
                for i, chat in enumerate(reversed(st.session_state.ai_assistant.conversation_history[-5:]), 1):
                    st.markdown(f"**Q{i} ({chat['timestamp']}):** {chat['question']}")
                    st.markdown(f"**A{i}:** {chat['response'][:200]}..." if len(chat['response']) > 200 else f"**A{i}:** {chat['response']}")
                    st.markdown("---")
        
        # Daily Breakdown Table
        st.header("📅 Daily Breakdown")
        st.dataframe(projections_df, use_container_width=True)

if __name__ == "__main__":
    main()