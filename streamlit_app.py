"""
Streamlit App - Integrated with Backend Pipeline
Entry point that connects dashboard with main.py backend
"""
import sys
from pathlib import Path
import streamlit as st
import time
from datetime import datetime
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import backend pipeline
from main import PipelineOrchestrator

# Import dashboard directly (bypass __init__.py issue)
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dashboard.app import MarketRiskDashboard


def check_data_status():
    """Check if data exists and is recent."""
    gold_dir = Path("data/gold")
    
    if not gold_dir.exists():
        return False, None, "No data directory found"
    
    # Check for prediction or feature files
    pred_files = list(gold_dir.glob("predictions_*.parquet"))
    feature_files = list(gold_dir.glob("features_*.parquet"))
    
    all_files = pred_files + feature_files
    
    if not all_files:
        return False, None, "No data files found"
    
    # Get latest file
    latest = max(all_files, key=lambda x: x.stat().st_mtime)
    mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
    
    # Check if fresh (less than 24 hours)
    age_hours = (datetime.now() - mod_time).total_seconds() / 3600
    is_fresh = age_hours < 24
    
    status = f"Last updated: {mod_time.strftime('%Y-%m-%d %H:%M')} ({age_hours:.1f}h ago)"
    
    return is_fresh, latest, status


def run_pipeline_background():
    """Run pipeline in background thread."""
    try:
        with st.spinner("Running pipeline... Please wait (2-3 minutes)"):
            # Create progress placeholder
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize orchestrator
            status_text.text("Initializing pipeline...")
            orchestrator = PipelineOrchestrator()
            progress_bar.progress(10)
            
            # Run scraping
            status_text.text("Step 1/4: Scraping market data...")
            scrape_success = orchestrator.run_scraping()
            progress_bar.progress(30)
            
            if not scrape_success:
                st.error("Scraping failed")
                return False
            
            # Run cleaning
            status_text.text("Step 2/4: Cleaning data...")
            clean_success = orchestrator.run_cleaning()
            progress_bar.progress(50)
            
            if not clean_success:
                st.error("Cleaning failed")
                return False
            
            # Run feature engineering
            status_text.text("Step 3/4: Engineering features...")
            feature_success = orchestrator.run_feature_engineering()
            progress_bar.progress(70)
            
            if not feature_success:
                st.error("Feature engineering failed")
                return False
            
            # Run model training
            status_text.text("Step 4/4: Training models...")
            predictions = orchestrator.run_model_training()
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if not predictions.empty:
                st.success("✓ Pipeline completed successfully!")
                return True
            else:
                st.warning("Pipeline completed but no predictions generated")
                return False
                
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return False


def main():
    """Main entry point - Dashboard with integrated backend."""
    
    # Page configuration
    st.set_page_config(
        page_title="Market Risk Intelligence",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'auto_run_done' not in st.session_state:
        st.session_state.auto_run_done = False
    
    # Header
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <h2 style='margin: 0; color: #2c3e50;'>Market Narrative Risk Intelligence System</h2>
        <p style='margin: 5px 0 0 0; color: #7f8c8d;'>Real-time market data analysis & risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check data status
    is_fresh, latest_file, status = check_data_status()
    
    # Control panel
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        if is_fresh:
            st.success(f"✓ {status}")
        else:
            st.warning(f"⚠ {status}")
    
    with col2:
        if st.button("Refresh Data", use_container_width=True, type="primary"):
            if run_pipeline_background():
                time.sleep(1)
                st.rerun()
    
    with col3:
        if st.button("Reload View", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Auto-run pipeline on first load if no data
    if not st.session_state.auto_run_done and not is_fresh:
        st.session_state.auto_run_done = True
        
        st.info("No fresh data found. Running pipeline automatically...")
        
        if run_pipeline_background():
            st.rerun()
    
    # Load and display dashboard
    try:
        dashboard = MarketRiskDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        
        # If no data, show setup instructions
        if not is_fresh or latest_file is None:
            st.markdown("""
            ### First Time Setup
            
            Click **"Refresh Data"** button above to:
            1. Scrape latest market data from Yahoo Finance
            2. Clean and process the data
            3. Train machine learning models
            4. Generate risk predictions
            
            This process takes 2-3 minutes.
            """)
            
            if st.button("Run Pipeline Now", type="primary", use_container_width=True):
                if run_pipeline_background():
                    st.rerun()


if __name__ == "__main__":
    main()
