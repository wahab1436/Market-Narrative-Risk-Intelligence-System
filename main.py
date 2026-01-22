"""
Streamlit App Launcher for Market Risk Intelligence Dashboard
Main entry point for Streamlit Cloud deployment.
"""
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the dashboard
if __name__ == "__main__":
    from src.dashboard.app import main
    main()
