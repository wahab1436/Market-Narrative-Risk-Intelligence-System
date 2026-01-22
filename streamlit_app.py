"""
Streamlit App - Dashboard with Pipeline Control
Professional interface for Market Narrative Risk Intelligence System
"""
import sys
from pathlib import Path
import streamlit as st
import subprocess
import time
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_data_status():
    """Check if data exists and when it was last updated."""
    gold_dir = Path("data/gold")
    
    if not gold_dir.exists():
        return None, "No data directory found"
    
    # Check for prediction files
    pred_files = list(gold_dir.glob("predictions_*.parquet"))
    if pred_files:
        latest = max(pred_files, key=lambda x: x.stat().st_mtime)
        mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
        return latest, f"Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Check for feature files
    feature_files = list(gold_dir.glob("features_*.parquet"))
    if feature_files:
        latest = max(feature_files, key=lambda x: x.stat().st_mtime)
        mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
        return latest, f"Features available: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    return None, "No data files found"

def run_pipeline():
    """Run the data pipeline."""
    try:
        st.info("Starting pipeline execution... This may take 2-3 minutes")
        
        # Run main.py pipeline
        with st.spinner("Executing pipeline..."):
            # Execute main.py
            result = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                st.success("Pipeline completed successfully")
                st.info("Refreshing dashboard with new data...")
                time.sleep(2)
                st.rerun()
            else:
                st.error(f"Pipeline failed with error code {result.returncode}")
                with st.expander("View Error Details"):
                    st.code(result.stderr)
                
    except subprocess.TimeoutExpired:
        st.error("Pipeline timeout exceeded 5 minutes. Please check system logs.")
    except Exception as e:
        st.error(f"Error executing pipeline: {e}")

def main():
    """Main entry point with dashboard and pipeline control."""
    
    # Set page config
    st.set_page_config(
        page_title="Market Risk Intelligence",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False
    
    # Check data status
    data_file, data_status = check_data_status()
    
    # Professional header
    st.markdown("""
    <div style='background-color: #2c3e50; padding: 20px; border-radius: 5px; margin-bottom: 25px;'>
        <h2 style='margin: 0; color: #ffffff; font-weight: 400;'>Market Narrative Risk Intelligence System</h2>
        <p style='margin: 5px 0 0 0; color: #ecf0f1; font-size: 0.95em;'>Real-time market analysis and risk assessment platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Control panel with professional styling
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style='padding: 10px; background-color: #f8f9fa; border-radius: 3px; border-left: 3px solid #3498db;'>
            <strong>System Status:</strong> {data_status}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("Refresh Data", use_container_width=True, type="primary"):
            run_pipeline()
    
    with col3:
        if st.button("Reload Dashboard", use_container_width=True):
            st.rerun()
    
    st.markdown("<div style='margin: 20px 0; border-top: 1px solid #e0e0e0;'></div>", unsafe_allow_html=True)
    
    # Load and run dashboard
    try:
        from src.dashboard.app import MarketRiskDashboard
        
        dashboard = MarketRiskDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"Dashboard initialization error: {e}")
        
        # If no data, offer to run pipeline
        if data_file is None:
            st.markdown("""
            <div style='padding: 20px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 3px; margin: 20px 0;'>
                <h4 style='margin-top: 0; color: #856404;'>Initial Setup Required</h4>
                <p style='color: #856404; margin-bottom: 0;'>No data available. The pipeline must be executed to generate initial data.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### Pipeline Execution Steps
            
            The data pipeline will perform the following operations:
            
            1. **Data Acquisition**: Scrape latest financial news from configured RSS sources
            2. **Data Processing**: Clean and validate scraped articles
            3. **Feature Engineering**: Extract and compute relevant features
            4. **Model Training**: Train machine learning models for risk assessment
            5. **Prediction Generation**: Generate predictions and insights
            
            **Estimated Time**: 2-3 minutes
            
            This is a one-time setup. Subsequent refreshes will use incremental updates.
            """)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Execute Pipeline", type="primary", use_container_width=True):
                    run_pipeline()

if __name__ == "__main__":
    main()
