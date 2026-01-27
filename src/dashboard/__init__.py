"""
Dashboard module initialization
"""

try:
    from src.dashboard.app import MarketRiskDashboard, main
except ImportError as e:
    # Fallback if app.py has issues
    import logging
    logging.warning(f"Dashboard import warning: {e}")
    MarketRiskDashboard = None
    main = None

__all__ = ['MarketRiskDashboard', 'main']
