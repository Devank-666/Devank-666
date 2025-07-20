#!/bin/bash

# Meta Ads Projection Application Startup Script
echo "🚀 Starting Meta Ads Performance Projections & AI Assistant..."
echo "📊 Loading application components..."

# Add local bin to PATH for streamlit
export PATH="/home/ubuntu/.local/bin:$PATH"

# Set environment variables for better performance
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Create directories if they don't exist
mkdir -p logs

# Run the application
echo "🌐 Starting Streamlit server..."
echo "📱 Once started, access the application at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

# Log startup time
echo "$(date): Starting Meta Ads Projection App" >> logs/startup.log

# Run streamlit with custom configuration
python3 -m streamlit run meta_ads_projection.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false