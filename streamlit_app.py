"""
Streamlit App - Dashboard with Pipeline Control
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
        st.info("🔄 Starting pipeline... This may take 2-3 minutes")
        
        # Create a placeholder for progress
        progress_placeholder = st.empty()
        
        # Run main.py pipeline
        with st.spinner("Running pipeline..."):
            # Execute main.py
            result = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                st.success("✅ Pipeline completed successfully!")
                st.info("🔄 Refreshing dashboard with new data...")
                time.sleep(2)
                st.rerun()
            else:
                st.error(f"❌ Pipeline failed with error code {result.returncode}")
                with st.expander("View Error Details"):
                    st.code(result.stderr)
                
    except subprocess.TimeoutExpired:
        st.error("⏱️ Pipeline timeout (>5 minutes). Please check logs.")
    except Exception as e:
        st.error(f"❌ Error running pipeline: {e}")

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
    
    # Top banner with pipeline control
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <h3 style='margin: 0; color: #2c3e50;'>Market Narrative Risk Intelligence System</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Control panel
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**Data Status:** {data_status}")
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            run_pipeline()
    
    with col3:
        if st.button("🔃 Reload Dashboard", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Load and run dashboard
    try:
        from src.dashboard.app import MarketRiskDashboard
        
        dashboard = MarketRiskDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        
        # If no data, offer to run pipeline
        if data_file is None:
            st.warning("⚠️ No data available. The pipeline needs to run first.")
            
            st.markdown("""
            ### First Time Setup
            
            Click the **"Refresh Data"** button above to:
            1. Scrape latest financial news (2-3 minutes)
            2. Process and clean the data
            3. Train machine learning models
            4. Generate predictions and insights
            
            This only needs to be done once, then you can refresh as needed.
            """)
            
            if st.button("▶️ Run Pipeline Now", type="primary", use_container_width=True):
                run_pipeline()

if __name__ == "__main__":
    main()
