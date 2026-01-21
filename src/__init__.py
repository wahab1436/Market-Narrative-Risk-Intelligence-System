"""
Market Narrative Risk Intelligence System
Avoid circular imports by only exporting names, not importing actual modules.
"""
__version__ = "1.0.0"
__author__ = "Market Intelligence Team"

# DO NOT import modules here that might cause circular imports
# Instead, export module names that can be imported directly

__all__ = [
    # Module names that can be imported
    'scraper',
    'preprocessing', 
    'models',
    'explainability',
    'dashboard',
    'utils',
    'eda'
]
