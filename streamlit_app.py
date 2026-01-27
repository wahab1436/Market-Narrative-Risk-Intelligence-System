"""
Streamlit App - Integrated with Backend Pipeline
Simplified version with better structure
"""
import streamlit as st
import time
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from main import PipelineOrchestrator
    from src.dashboard.app import MarketRiskDashboard
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Make sure the backend pipeline is properly set up.")


def check_data_status():
    """Check if data exists and is recent."""
    gold_dir = Path("data/gold")
    
    if not gold_dir.exists():
        return False, None, "No data directory found"
    
    # Check for data files
    files = list(gold_dir.glob("*.parquet"))
    if not files:
        return False, None, "No data files found"
    
    # Get latest file
    latest = max(files, key=lambda x: x.stat().st_mtime)
    mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
    age_hours = (datetime.now() - mod_time).total_seconds() / 3600
    
    is_fresh = age_hours < 24
    status = f"Last updated: {mod_time.strftime('%Y-%m-%d %H:%M')}"
    
    return is_fresh, latest, status


def run_pipeline():
    """Run the pipeline with progress tracking."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        orchestrator = PipelineOrchestrator()
        
        # Define pipeline steps
        steps = [
            ("Initializing", 10, None),
            ("Scraping data", 30, orchestrator.run_scraping),
            ("Cleaning data", 50, orchestrator.run_cleaning),
            ("Engineering features", 70, orchestrator.run_feature_engineering),
            ("Training models", 90, orchestrator.run_model_training),
            ("Finalizing", 100, None)
        ]
        
        for step_name, progress, func in steps:
            status_text.text(f"Step: {step_name}")
            progress_bar.progress(progress)
            
            if func:
                if not func():
                    st.error(f"{step_name} failed")
                    return False
            
            time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        st.success("Pipeline completed successfully!")
        return True
        
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        return False


def main():
    """Main entry point."""
    st.set_page_config(
        page_title="Market Risk Intelligence",
        layout="wide"
    )
    
    # Initialize session state
    if 'auto_run' not in st.session_state:
        st.session_state.auto_run = False
    
    # Header
    st.title("Market Narrative Risk Intelligence System")
    st.subheader("Real-time market data analysis & risk assessment")
    st.divider()
    
    # Check data status
    is_fresh, _, status_msg = check_data_status()
    
    # Control panel
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if is_fresh:
            st.success(status_msg)
        else:
            st.warning(status_msg)
    
    with col2:
        if st.button("Refresh Data", type="primary"):
            if run_pipeline():
                time.sleep(1)
                st.rerun()
    
    st.divider()
    
    # Auto-run if no fresh data
    if not st.session_state.auto_run and not is_fresh:
        st.session_state.auto_run = True
        st.info("No fresh data found. Running pipeline...")
        
        if run_pipeline():
            st.rerun()
    
    # Load dashboard
    try:
        dashboard = MarketRiskDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")
        
        if not is_fresh:
            st.info("Click 'Refresh Data' button to run the pipeline and generate data.")


if __name__ == "__main__":
    main()
