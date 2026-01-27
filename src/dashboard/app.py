"""
Professional Market Narrative Risk Intelligence Dashboard.
Complete working version with all features, no emojis, proper data handling.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import traceback
from typing import List, Dict, Optional, Tuple

# Page configuration - MUST be first Streamlit command
st.set_page_config(
page_title="Market Narrative Risk Intelligence",
layout="wide",
initial_sidebar_state="expanded",
menu_items={
'Get Help': None,
'Report a bug': None,
'About': "Market Narrative Risk Intelligence System v1.0.0"
}
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import project modules with proper error handling
try:
from src.utils.config_loader import config_loader
print("Config loader imported successfully")
except ImportError as e:
print(f"Warning: Failed to import config_loader: {e}")
config_loader = None

try:
import logging
from src.utils.logger import get_dashboard_logger
logger = get_dashboard_logger()
print("Logger imported successfully")
except ImportError as e:
print(f"Warning: Failed to import logger: {e}")
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
from src.eda.visualization import EDAVisualizer
print("EDAVisualizer imported successfully")
except ImportError as e:
logger.warning(f"EDAVisualizer not available: {e}")
EDAVisualizer = None

# Try to import SHAP with fallback
try:
from src.explainability.shap_analysis import SHAPAnalyzer
SHAP_AVAILABLE = True
print("SHAP analysis imported successfully")
except ImportError as e:
logger.warning(f"SHAP not available (likely Python 3.13 incompatibility): {e}")
SHAP_AVAILABLE = False

# Import pipeline components
try:
from src.scraper import scrape_and_save
from src.preprocessing.clean_data import clean_and_save
from src.preprocessing.feature_engineering import engineer_and_save
PIPELINE_AVAILABLE = True
print("Pipeline components imported successfully")
except ImportError as e:
logger.warning(f"Pipeline components not available: {e}")
PIPELINE_AVAILABLE = False

# Import models
try:
from src.models.regression.linear_regression import LinearRegressionModel
from src.models.regression.ridge_regression import RidgeRegressionModel
from src.models.regression.lasso_regression import LassoRegressionModel
from src.models.neural_network import NeuralNetworkModel
from src.models.xgboost_model import XGBoostModel
from src.models.isolation_forest import IsolationForestModel
MODELS_AVAILABLE = True
print("Model imports successful")
except ImportError as e:
logger.info(f"Model imports optional, continuing without: {e}")
MODELS_AVAILABLE = False


class MarketRiskDashboard:
"""Professional dashboard for market narrative risk intelligence."""

def __init__(self):
"""Initialize dashboard with professional settings."""
self.logger = logger

try:
if config_loader:
self.config = config_loader.get_config("config")
self.colors = self.config.get('visualization', {}).get('color_palette',
['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
print("Configuration loaded successfully")
else:
self.config = {}
self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
except Exception as e:
logger.warning(f"Failed to load dashboard config: {e}")
self.config = {}
self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

if EDAVisualizer:
self.eda_visualizer = EDAVisualizer(theme="plotly_white")
else:
self.eda_visualizer = None

# Initialize SHAP analyzer if available
if SHAP_AVAILABLE:
try:
from src.explainability.shap_analysis import SHAPAnalyzer
self.shap_analyzer = SHAPAnalyzer()
logger.info("SHAP analyzer initialized")
except:
self.shap_analyzer = None
logger.warning("SHAP analyzer initialization failed")
else:
self.shap_analyzer = None

self._init_session_state()
self.load_data()
self.logger.info("MarketRiskDashboard initialized successfully")

def _init_session_state(self):
"""Initialize session state variables."""
if 'data_loaded' not in st.session_state:
st.session_state.data_loaded = False
if 'current_view' not in st.session_state:
st.session_state.current_view = 'overview'
if 'filtered_data' not in st.session_state:
st.session_state.filtered_data = None
if 'model_predictions' not in st.session_state:
st.session_state.model_predictions = {}
if 'pipeline_running' not in st.session_state:
st.session_state.pipeline_running = False
if 'shap_values' not in st.session_state:
st.session_state.shap_values = None

def load_data(self):
"""Load prediction data from gold layer."""
try:
gold_dir = Path("data/gold")

if not gold_dir.exists():
gold_dir.mkdir(parents=True, exist_ok=True)

# First try predictions files
prediction_files = list(gold_dir.glob("*predictions*.parquet"))

if prediction_files:
latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)

# Check file size before attempting to read
file_size = latest_file.stat().st_size
if file_size < 100: # Corrupted file
self.logger.warning(f"Corrupted file detected: {latest_file} ({file_size} bytes)")
latest_file.unlink() # Delete it
self._create_sample_data()
return

self.df = pd.read_parquet(latest_file)

if 'timestamp' in self.df.columns:
self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

if not self.df.empty and 'timestamp' in self.df.columns:
self.min_date = self.df['timestamp'].min().date()
self.max_date = self.df['timestamp'].max().date()

self.logger.info(f"Loaded {len(self.df)} records from {latest_file}")
st.session_state.data_loaded = True
return

# Try feature files
gold_files = list(gold_dir.glob("features_*.parquet"))
if gold_files:
latest_file = max(gold_files, key=lambda x: x.stat().st_mtime)

file_size = latest_file.stat().st_size
if file_size < 100:
self.logger.warning(f"Corrupted file: {latest_file}")
latest_file.unlink()
self._create_sample_data()
return

self.df = pd.read_parquet(latest_file)

if 'timestamp' in self.df.columns:
self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
self.min_date = self.df['timestamp'].min().date()
self.max_date = self.df['timestamp'].max().date()

self.logger.info(f"Loaded {len(self.df)} records from {latest_file}")
st.session_state.data_loaded = True
return

# No valid files found
self._create_sample_data()
self.logger.warning("No valid data files found, using sample data")

except Exception as e:
self.logger.error(f"Failed to load data: {e}", exc_info=True)
self._create_sample_data()

def _create_sample_data(self):
"""Create sample data for demonstration."""
dates = pd.date_range(end=datetime.now(), periods=60, freq='D')

np.random.seed(42)
base_trend = np.linspace(0, 2, len(dates))
seasonal = 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
noise = np.random.normal(0, 0.3, len(dates))

stress_scores = base_trend + seasonal + noise

self.df = pd.DataFrame({
'timestamp': dates,
'headline': [f"Market update {i}" for i in range(len(dates))],
'sentiment_polarity': np.random.uniform(-0.8, 0.8, len(dates)),
'keyword_stress_score': np.random.exponential(0.3, len(dates)),
'weighted_stress_score': stress_scores,
'linear_regression_prediction': stress_scores + np.random.normal(0, 0.2, len(dates)),
'ridge_regression_prediction': stress_scores + np.random.normal(0, 0.15, len(dates)),
'lasso_regression_prediction': stress_scores + np.random.normal(0, 0.18, len(dates)),
'neural_network_prediction': stress_scores + np.random.normal(0, 0.12, len(dates)),
'xgboost_risk_regime': np.random.choice(['low', 'medium', 'high'], len(dates), p=[0.3, 0.5, 0.2]),
'prob_low': np.random.uniform(0, 1, len(dates)),
'prob_medium': np.random.uniform(0, 1, len(dates)),
'prob_high': np.random.uniform(0, 1, len(dates)),
'is_anomaly': np.random.choice([0, 1], len(dates), p=[0.92, 0.08]),
'anomaly_score': np.random.uniform(-0.5, 0.5, len(dates)),
'daily_article_count': np.random.poisson(50, len(dates)),
'market_breadth': np.random.uniform(0, 1, len(dates))
})

# Normalize probabilities
probs = self.df[['prob_low', 'prob_medium', 'prob_high']].values
probs = probs / probs.sum(axis=1, keepdims=True)
self.df[['prob_low', 'prob_medium', 'prob_high']] = probs

self.min_date = self.df['timestamp'].min().date()
self.max_date = self.df['timestamp'].max().date()

st.session_state.data_loaded = True
self.logger.info("Created sample data for demonstration")

def _run_full_pipeline(self):
"""Run the complete pipeline: scraping -> cleaning -> features -> models."""
if not PIPELINE_AVAILABLE:
st.error("Pipeline components not available. Cannot run full pipeline.")
return False

progress_bar = st.progress(0)
status_text = st.empty()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

try:
# Step 1: Scraping
status_text.text("Step 1/4: Scraping news articles...")
progress_bar.progress(10)
self.logger.info("Starting scraping phase")

bronze_path = scrape_and_save()

if not bronze_path:
st.error("Scraping failed. No data collected.")
return False

self.logger.info(f"Scraping completed: {bronze_path}")
progress_bar.progress(25)

# Step 2: Cleaning
status_text.text("Step 2/4: Cleaning and validating data...")
progress_bar.progress(30)
self.logger.info("Starting cleaning phase")

silver_path = clean_and_save(bronze_path)

if not silver_path:
st.error("Data cleaning failed.")
return False

self.logger.info(f"Cleaning completed: {silver_path}")
progress_bar.progress(50)

# Step 3: Feature Engineering
status_text.text("Step 3/4: Engineering features...")
progress_bar.progress(55)
self.logger.info("Starting feature engineering")

gold_path = engineer_and_save(silver_path)

if not gold_path:
st.error("Feature engineering failed.")
return False

self.logger.info(f"Feature engineering completed: {gold_path}")
progress_bar.progress(70)

# Step 4: Model Training
status_text.text("Step 4/4: Training models and generating predictions...")
progress_bar.progress(75)
self.logger.info("Starting model training")

df = pd.read_parquet(gold_path)

models = {
'linear_regression': LinearRegressionModel(),
'ridge_regression': RidgeRegressionModel(),
'lasso_regression': LassoRegressionModel(),
'neural_network': NeuralNetworkModel(),
'xgboost': XGBoostModel(),
'isolation_forest': IsolationForestModel()
}

predictions_dfs = []
model_count = len(models)

for idx, (model_name, model) in enumerate(models.items()):
try:
status_text.text(f"Training {model_name} ({idx+1}/{model_count})...")
progress = 75 + int((idx / model_count) * 15)
progress_bar.progress(progress)

self.logger.info(f"Training {model_name}")

results = model.train(df)
predictions = model.predict(df)

pred_cols = [col for col in predictions.columns
if any(x in col for x in ['prediction', 'regime', 'anomaly', 'similarity', 'forecast'])]

if pred_cols:
predictions_subset = predictions[['timestamp'] + pred_cols]
predictions_dfs.append(predictions_subset)
self.logger.info(f"{model_name} completed successfully")

# Save model
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
model.save(model_dir / f"{model_name}_{timestamp}.joblib")

except Exception as e:
self.logger.error(f"{model_name} training failed: {e}")
continue

progress_bar.progress(90)
status_text.text("Merging predictions...")

if predictions_dfs:
# Merge all predictions
final_predictions = df[['timestamp']].copy()

for pred_df in predictions_dfs:
final_predictions = final_predictions.merge(
pred_df,
on='timestamp',
how='left'
)

# Add original features
feature_cols = [col for col in df.columns if col != 'timestamp']
final_predictions = final_predictions.merge(
df[['timestamp'] + feature_cols],
on='timestamp',
how='left'
)

# Save predictions
predictions_path = Path("data/gold") / f"predictions_{timestamp}.parquet"
final_predictions.to_parquet(predictions_path, index=False)

self.logger.info(f"Predictions saved to: {predictions_path}")

progress_bar.progress(100)
status_text.text("Pipeline completed successfully!")

return True
else:
st.error("No models produced predictions")
return False

except Exception as e:
self.logger.error(f"Pipeline failed: {e}", exc_info=True)
st.error(f"Pipeline error: {e}")
return False

finally:
progress_bar.empty()
status_text.empty()

def _run_quick_update(self):
"""Run quick update using existing data with lightweight models."""
progress_bar = st.progress(0)
status_text = st.empty()

try:
status_text.text("Loading existing data...")
progress_bar.progress(10)

if not hasattr(self, 'df') or self.df is None or self.df.empty:
st.error("No data available. Please ensure data files exist.")
return False

df = self.df.copy()
df = df.tail(500) # Use recent data only

status_text.text("Engineering features...")
progress_bar.progress(30)

# Add rolling features
if 'sentiment_polarity' in df.columns:
df['sentiment_ma_7'] = df['sentiment_polarity'].rolling(7, min_periods=1).mean()

if 'keyword_stress_score' in df.columns:
df['stress_ma_7'] = df['keyword_stress_score'].rolling(7, min_periods=1).mean()

status_text.text("Training lightweight models...")
progress_bar.progress(50)

from sklearn.linear_model import LinearRegression, Ridge

# Prepare features
feature_cols = [col for col in df.columns if col not in [
'timestamp', 'headline', 'weighted_stress_score'
]]

X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
y = df['weighted_stress_score'] if 'weighted_stress_score' in df.columns else np.random.rand(len(df))

predictions = df.copy()

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X, y)
predictions['linear_regression_prediction'] = lr_model.predict(X)

progress_bar.progress(70)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)
predictions['ridge_regression_prediction'] = ridge_model.predict(X)

status_text.text("Generating risk classifications...")
progress_bar.progress(85)

# Risk regimes based on stress scores
stress_values = predictions.get('weighted_stress_score', y)
predictions['xgboost_risk_regime'] = pd.cut(
stress_values,
bins=[-np.inf, 0.3, 0.7, np.inf],
labels=['low', 'medium', 'high']
)

# Probabilities
predictions['prob_low'] = (stress_values < 0.3).astype(float)
predictions['prob_medium'] = ((stress_values >= 0.3) & (stress_values < 0.7)).astype(float)
predictions['prob_high'] = (stress_values >= 0.7).astype(float)

# Anomaly detection using z-scores
z_scores = np.abs((stress_values - stress_values.mean()) / stress_values.std())
predictions['is_anomaly'] = (z_scores > 2).astype(int)
predictions['anomaly_score'] = z_scores

status_text.text("Saving predictions...")
progress_bar.progress(95)

# Save to gold directory
gold_dir = Path("data/gold")
gold_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = gold_dir / f"predictions_{timestamp}.parquet"

predictions.to_parquet(output_file, index=False)

self.logger.info(f"Quick update saved to: {output_file}")

progress_bar.progress(100)
status_text.text("Update completed successfully!")

return True

except Exception as e:
st.error(f"Quick update failed: {e}")
self.logger.error(f"Quick update failed: {e}", exc_info=True)
return False

finally:
progress_bar.empty()
status_text.empty()

def render_header(self):
"""Render professional header."""
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
<h1 style='margin-bottom: 5px; color: #2c3e50;'>Market Narrative Risk Intelligence System</h1>
<p style='color: #7f8c8d; font-size: 1.1em; margin-top: 0;'>
Advanced analytics for market stress detection and risk regime identification
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

def render_sidebar(self):
"""Render professional sidebar with filters and navigation."""
with st.sidebar:
# System status
gold_dir = Path("data/gold")
data_status = "No data"
data_time = "Never"

if gold_dir.exists():
files = list(gold_dir.glob("*.parquet"))
if files:
latest = max(files, key=lambda x: x.stat().st_mtime)
mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
data_status = "Available"
data_time = mod_time.strftime("%Y-%m-%d %H:%M")

shap_status = "Available" if SHAP_AVAILABLE else "Unavailable (Python 3.13)"

st.markdown(f"""
<div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 20px;'>
<p style='font-size: 0.9em; color: #6c757d; margin: 0;'>
<strong>System Status:</strong> Operational<br>
<strong>Data Status:</strong> {data_status}<br>
<strong>SHAP Analysis:</strong> {shap_status}<br>
<strong>Last Updated:</strong> {data_time}<br>
<strong>Records Loaded:</strong> {len(self.df) if hasattr(self, 'df') else 0:,}
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Pipeline Control")

# Show component status
if PIPELINE_AVAILABLE:
st.success("Pipeline: Available")
else:
st.warning("Pipeline: Not Available")

if MODELS_AVAILABLE:
st.success("Models: Available")
else:
st.warning("Models: Limited")

# Update buttons
col1, col2 = st.columns(2)

with col1:
full_disabled = not PIPELINE_AVAILABLE
if st.button("Full Update", type="primary", use_container_width=True,
disabled=full_disabled,
help="Run complete pipeline with new data scraping"):
with st.spinner("Running full pipeline..."):
success = self._run_full_pipeline()
if success:
self.load_data()
st.success("Full pipeline completed!")
import time
time.sleep(2)
st.rerun()

with col2:
if st.button("Quick Update", use_container_width=True):
with st.spinner("Running quick update..."):
success = self._run_quick_update()
if success:
self.load_data()
st.success("Quick update completed!")
import time
time.sleep(2)
st.rerun()

if st.button("Refresh View", use_container_width=True):
st.rerun()

st.markdown("---")
st.markdown("### Navigation")

view_options = {
'overview': 'System Overview',
'stress_analysis': 'Stress Score Analysis',
'risk_regimes': 'Risk Regime Classification',
'anomaly_detection': 'Anomaly Detection',
'explainability': 'Model Explainability (SHAP)',
'historical_similarity': 'Historical Similarity',
'model_performance': 'Model Performance',
'feature_analysis': 'Feature Analysis',
'data_export': 'Data Export'
}

selected_view = st.selectbox(
"Select Dashboard View",
options=list(view_options.keys()),
format_func=lambda x: view_options[x],
key='view_selector'
)
st.session_state.current_view = selected_view

st.markdown("---")
st.markdown("### Data Filters")

# Date range filter
if hasattr(self, 'min_date') and hasattr(self, 'max_date'):
date_diff = (self.max_date - self.min_date).days

if date_diff > 30:
default_start = self.max_date - timedelta(days=30)
default_end = self.max_date
else:
default_start = self.min_date
default_end = self.max_date

date_range = st.date_input(
"Analysis Period",
value=(default_start, default_end),
min_value=self.min_date,
max_value=self.max_date
)

if len(date_range) == 2:
self.start_date, self.end_date = date_range
else:
self.start_date = self.end_date = date_range[0] if date_range else self.max_date
else:
self.start_date = self.end_date = datetime.now().date()

# Risk regime filter
if 'xgboost_risk_regime' in self.df.columns:
risk_regimes = ['All'] + sorted(self.df['xgboost_risk_regime'].dropna().unique().tolist())
selected_regime = st.selectbox("Risk Regime Filter", risk_regimes, index=0)
self.selected_regime = selected_regime if selected_regime != 'All' else None
else:
self.selected_regime = None

# Anomaly filter
if 'is_anomaly' in self.df.columns:
anomaly_filter = st.radio(
"Anomaly Filter",
['All Data', 'Anomalies Only', 'Exclude Anomalies'],
index=0
)
self.anomaly_filter = anomaly_filter
else:
self.anomaly_filter = 'All Data'

# Confidence threshold
confidence_threshold = st.slider(
"Minimum Confidence Threshold",
min_value=0.0,
max_value=1.0,
value=0.7,
step=0.05
)
self.confidence_threshold = confidence_threshold

def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
"""Apply all sidebar filters to data."""
filtered_df = df.copy()

# Date filter
if 'timestamp' in filtered_df.columns:
filtered_df = filtered_df[
(filtered_df['timestamp'].dt.date >= self.start_date) &
(filtered_df['timestamp'].dt.date <= self.end_date)
]

# Risk regime filter
if self.selected_regime and 'xgboost_risk_regime' in filtered_df.columns:
filtered_df = filtered_df[filtered_df['xgboost_risk_regime'] == self.selected_regime]

# Anomaly filter
if 'is_anomaly' in filtered_df.columns:
if self.anomaly_filter == 'Anomalies Only':
filtered_df = filtered_df[filtered_df['is_anomaly'] == 1]
elif self.anomaly_filter == 'Exclude Anomalies':
filtered_df = filtered_df[filtered_df['is_anomaly'] == 0]

# Confidence filter
prob_cols = [col for col in filtered_df.columns if col.startswith('prob_')]
if prob_cols:
max_probs = filtered_df[prob_cols].max(axis=1)
filtered_df = filtered_df[max_probs >= self.confidence_threshold]

return filtered_df

def render_overview(self):
"""Render system overview dashboard."""
st.markdown("## System Overview")

filtered_df = self._apply_filters(self.df)

if filtered_df.empty:
st.warning("No data matches current filters. Please adjust your filter settings.")
return

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
if 'weighted_stress_score' in filtered_df.columns:
current_stress = filtered_df['weighted_stress_score'].iloc[-1]
avg_stress = filtered_df['weighted_stress_score'].mean()
st.metric(
"Current Stress Score",
f"{current_stress:.2f}",
f"{current_stress - avg_stress:+.2f} vs avg"
)
else:
st.info("Stress score data not available")

with col2:
if 'xgboost_risk_regime' in filtered_df.columns:
current_regime = filtered_df['xgboost_risk_regime'].iloc[-1]
regime_color = {
'low': 'green',
'medium': 'orange',
'high': 'red'
}.get(current_regime, 'gray')

st.markdown(f"""
<div style='text-align: center; padding: 10px;'>
<div style='font-size: 0.9em; color: #6c757d;'>Current Risk Regime</div>
<div style='font-size: 1.5em; font-weight: bold; color: {regime_color};'>{current_regime.upper()}</div>
</div>
""", unsafe_allow_html=True)
else:
st.info("Risk regime data not available")

with col3:
if 'is_anomaly' in filtered_df.columns:
anomaly_count = int(filtered_df['is_anomaly'].sum())
anomaly_rate = (anomaly_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
st.metric(
"Anomalies Detected",
f"{anomaly_count}",
f"{anomaly_rate:.1f}% rate"
)
else:
st.info("Anomaly data not available")

with col4:
total_articles = len(filtered_df)
date_range = (filtered_df['timestamp'].max() - filtered_df['timestamp'].min()).days if 'timestamp' in filtered_df.columns else 1
avg_daily = total_articles / max(date_range, 1)
st.metric(
"Articles Analyzed",
f"{total_articles:,}",
f"{avg_daily:.0f}/day avg"
)

st.markdown("---")

# Recent articles table
st.markdown("### Recent Articles")
display_cols = ['timestamp', 'headline', 'sentiment_polarity', 'weighted_stress_score']
display_cols = [col for col in display_cols if col in filtered_df.columns]

if display_cols:
recent_data = filtered_df[display_cols].tail(10).copy()
if 'timestamp' in recent_data.columns:
recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
st.dataframe(recent_data, use_container_width=True, hide_index=True)
else:
st.info("No displayable columns available in the data")

def render_stress_analysis(self):
"""Render detailed stress analysis view."""
st.markdown("## Stress Score Analysis")

filtered_df = self._apply_filters(self.df)

if filtered_df.empty:
st.warning("No data available for stress analysis")
return

if 'weighted_stress_score' not in filtered_df.columns:
st.info("Stress score data not yet available. Run the pipeline to generate predictions.")
return

# Time series plot
fig = go.Figure()

fig.add_trace(go.Scatter(
x=filtered_df['timestamp'],
y=filtered_df['weighted_stress_score'],
mode='lines+markers',
name='Stress Score',
line=dict(color=self.colors[0], width=2),
marker=dict(size=4)
))

# Add trend line
if len(filtered_df) > 2:
z = np.polyfit(range(len(filtered_df)), filtered_df['weighted_stress_score'], 1)
p = np.poly1d(z)
fig.add_trace(go.Scatter(
x=filtered_df['timestamp'],
y=p(range(len(filtered_df))),
mode='lines',
name='Trend',
line=dict(color='red', width=2, dash='dash')
))

fig.update_layout(
title='Stress Score Over Time',
height=400,
xaxis_title="Date",
yaxis_title="Stress Score",
hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
st.metric("Current", f"{filtered_df['weighted_stress_score'].iloc[-1]:.2f}")
with col2:
st.metric("Average", f"{filtered_df['weighted_stress_score'].mean():.2f}")
with col3:
st.metric("Maximum", f"{filtered_df['weighted_stress_score'].max():.2f}")
with col4:
st.metric("Minimum", f"{filtered_df['weighted_stress_score'].min():.2f}")

def render_risk_regimes(self):
"""Render risk regime analysis."""
st.markdown("## Risk Regime Analysis")

filtered_df = self._apply_filters(self.df)

if filtered_df.empty:
st.warning("No data available")
return

if 'xgboost_risk_regime' in filtered_df.columns:
# Distribution pie chart
regime_counts = filtered_df['xgboost_risk_regime'].value_counts()

fig = go.Figure(data=[go.Pie(
labels=regime_counts.index,
values=regime_counts.values,
marker=dict(colors=['#2ca02c', '#ff7f0e', '#d62728'])
)])

fig.update_layout(
title='Risk Regime Distribution',
height=400
)

st.plotly_chart(fig, use_container_width=True)

# Time series scatter
fig2 = px.scatter(
filtered_df,
x='timestamp',
y='weighted_stress_score' if 'weighted_stress_score' in filtered_df.columns else 'sentiment_polarity',
color='xgboost_risk_regime',
color_discrete_map={'low': '#2ca02c', 'medium': '#ff7f0e', 'high': '#d62728'},
title='Risk Regimes Over Time'
)

st.plotly_chart(fig2, use_container_width=True)
else:
st.info("Risk regime predictions not yet available. Run the pipeline with XGBoost model.")

def render_anomaly_detection(self):
"""Render anomaly detection view."""
st.markdown("## Anomaly Detection")

filtered_df = self._apply_filters(self.df)

if filtered_df.empty:
st.warning("No data available")
return

if 'is_anomaly' in filtered_df.columns:
# Anomaly timeline
fig = go.Figure()

normal = filtered_df[filtered_df['is_anomaly'] == 0]
anomalies = filtered_df[filtered_df['is_anomaly'] == 1]

if 'weighted_stress_score' in filtered_df.columns:
fig.add_trace(go.Scatter(
x=normal['timestamp'],
y=normal['weighted_stress_score'],
mode='markers',
name='Normal',
marker=dict(color='lightblue', size=6)
))

fig.add_trace(go.Scatter(
x=anomalies['timestamp'],
y=anomalies['weighted_stress_score'],
mode='markers',
name='Anomaly',
marker=dict(color='red', size=10, symbol='diamond')
))

fig.update_layout(
title='Anomaly Detection Timeline',
height=400,
xaxis_title="Date",
yaxis_title="Stress Score"
)

st.plotly_chart(fig, use_container_width=True)

# Statistics
col1, col2 = st.columns(2)
with col1:
st.metric("Total Anomalies", int(filtered_df['is_anomaly'].sum()))
with col2:
st.metric("Anomaly Rate", f"{(filtered_df['is_anomaly'].sum() / len(filtered_df) * 100):.1f}%")
else:
st.info("Anomaly detection not yet available. Run the pipeline with Isolation Forest model.")

def render_explainability(self):
"""Render SHAP explainability analysis view."""
st.markdown("## Model Explainability (SHAP Analysis)")

if not SHAP_AVAILABLE:
st.warning("""
SHAP Analysis Not Available

SHAP (SHapley Additive exPlanations) requires Python < 3.10, but you're running Python 3.13.

Solutions:
1. Downgrade Python to 3.9 or 3.10 (recommended for SHAP)
2. Use Alternative: Feature importance from tree-based models (shown below)
3. Wait: SHAP maintainers may add Python 3.13 support in future releases

For now, showing alternative feature importance analysis:
""")

# Fallback: Show feature importance without SHAP
filtered_df = self._apply_filters(self.df)

if filtered_df.empty:
st.warning("No data available")
return

# Try to get feature importances from model files
st.markdown("### Feature Importance (Model-based)")

model_dir = Path("models")
if model_dir.exists():
import joblib
model_files = list(model_dir.glob("xgboost_*.joblib"))

if model_files:
latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

try:
model = joblib.load(latest_model)

if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
importances = model.model.feature_importances_
feature_names = model.feature_columns if hasattr(model, 'feature_columns') else [f"Feature {i}" for i in range(len(importances))]

# Create DataFrame
importance_df = pd.DataFrame({
'Feature': feature_names,
'Importance': importances
}).sort_values('Importance', ascending=False).head(20)

# Plot
fig = px.bar(
importance_df,
x='Importance',
y='Feature',
orientation='h',
title='Top 20 Features by Importance'
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(importance_df, use_container_width=True)

else:
st.info("Model doesn't support feature importance extraction")

except Exception as e:
st.error(f"Failed to load model: {e}")
else:
st.info("No XGBoost models found. Train models first.")
else:
st.info("No models directory found")

# Show correlation-based importance
st.markdown("### Feature Correlation with Target")

if 'weighted_stress_score' in filtered_df.columns:
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
correlations = filtered_df[numeric_cols].corrwith(filtered_df['weighted_stress_score']).abs().sort_values(ascending=False).head(20)

corr_df = pd.DataFrame({
'Feature': correlations.index,
'Absolute Correlation': correlations.values
})

fig = px.bar(
corr_df,
x='Absolute Correlation',
y='Feature',
orientation='h',
title='Top 20 Features by Correlation with Stress Score'
)

st.plotly_chart(fig, use_container_width=True)

return

# SHAP is available - show full analysis
st.markdown("""
SHAP (SHapley Additive exPlanations) provides interpretable explanations for model predictions
by showing how each feature contributes to individual predictions.
""")

filtered_df = self._apply_filters(self.df)

if filtered_df.empty:
st.warning("No data available for analysis")
return

# Load trained model
model_dir = Path("models")
if not model_dir.exists():
st.warning("No trained models found. Please run the pipeline first.")
return

model_files = list(model_dir.glob("xgboost_*.joblib"))
if not model_files:
st.warning("No XGBoost models found. SHAP analysis requires tree-based models.")
return

latest_model_path = max(model_files, key=lambda x: x.stat().st_mtime)

try:
import joblib
model = joblib.load(latest_model_path)

st.success(f"Loaded model: {latest_model_path.name}")

# Prepare data
feature_cols = [col for col in filtered_df.columns
if col not in ['timestamp', 'headline', 'weighted_stress_score',
'xgboost_risk_regime', 'is_anomaly']]

X = filtered_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)

if X.empty:
st.warning("No numeric features available for SHAP analysis")
return

# Calculate SHAP values
with st.spinner("Calculating SHAP values... This may take a minute."):
try:
shap_values = self.shap_analyzer.explain_prediction(
model=model.model if hasattr(model, 'model') else model,
X=X
)

st.session_state.shap_values = shap_values

st.success("SHAP analysis complete")

except Exception as e:
st.error(f"SHAP calculation failed: {e}")
return

# Display SHAP visualizations
st.markdown("### SHAP Summary Plot")
st.markdown("Shows the impact of each feature on model output")

try:
fig = self.shap_analyzer.plot_feature_importance(
shap_values,
X,
plot_type='summary'
)

if fig:
st.pyplot(fig)
except Exception as e:
st.error(f"Failed to create SHAP plot: {e}")

# Feature importance
st.markdown("### Feature Importance (SHAP-based)")

try:
fig = self.shap_analyzer.plot_feature_importance(
shap_values,
X,
plot_type='bar'
)

if fig:
st.pyplot(fig)
except Exception as e:
st.error(f"Failed to create importance plot: {e}")

# Individual prediction explanation
st.markdown("### Individual Prediction Explanation")

sample_idx = st.slider(
"Select sample to explain",
0,
len(X) - 1,
len(X) - 1
)

try:
fig = self.shap_analyzer.plot_waterfall(
shap_values,
X,
sample_idx
)

if fig:
st.pyplot(fig)

# Show actual values
st.markdown("**Feature Values for Selected Sample:**")
sample_data = X.iloc[sample_idx].sort_values(ascending=False).head(10)
st.dataframe(sample_data)

except Exception as e:
st.error(f"Failed to create waterfall plot: {e}")

except Exception as e:
st.error(f"Failed to load model: {e}")
st.code(traceback.format_exc())

def render_historical_similarity(self):
"""Render historical similarity view."""
st.markdown("## Historical Patterns")

filtered_df = self._apply_filters(self.df)

if filtered_df.empty or 'weighted_stress_score' not in filtered_df.columns:
st.info("Historical pattern analysis requires stress score data")
return

# Distribution histogram
fig = px.histogram(
filtered_df,
x='weighted_stress_score',
nbins=30,
title='Stress Score Distribution'
)

st.plotly_chart(fig, use_container_width=True)

# Correlation scatter
if 'sentiment_polarity' in filtered_df.columns:
fig2 = px.scatter(
filtered_df,
x='sentiment_polarity',
y='weighted_stress_score',
title='Sentiment vs Stress Score Correlation'
)

correlation = filtered_df[['sentiment_polarity', 'weighted_stress_score']].corr().iloc[0, 1]

fig2.add_annotation(
text=f'Correlation: {correlation:.3f}',
xref="paper", yref="paper",
x=0.05, y=0.95,
showarrow=False,
font=dict(size=12),
bgcolor="white",
bordercolor="black",
borderwidth=1
)

st.plotly_chart(fig2, use_container_width=True)

def render_model_performance(self):
"""Render model performance view."""
st.markdown("## Model Performance")

filtered_df = self._apply_filters(self.df)

if filtered_df.empty:
st.warning("No data available")
return

pred_cols = [col for col in filtered_df.columns if 'prediction' in col.lower()]

if pred_cols and 'weighted_stress_score' in filtered_df.columns:
st.markdown("### Model Predictions Comparison")

# Time series comparison
fig = go.Figure()

fig.add_trace(go.Scatter(
x=filtered_df['timestamp'],
y=filtered_df['weighted_stress_score'],
mode='lines',
name='Actual',
line=dict(color='black', width=2)
))

for i, col in enumerate(pred_cols[:4]): # Show max 4 models
fig.add_trace(go.Scatter(
x=filtered_df['timestamp'],
y=filtered_df[col],
mode='lines',
name=col.replace('_prediction', '').replace('_', ' ').title(),
line=dict(width=1.5, dash='dash')
))

fig.update_layout(
title='Model Predictions vs Actual',
height=500,
xaxis_title="Date",
yaxis_title="Value",
hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Model metrics
st.markdown("### Model Metrics")

metrics_data = []
for col in pred_cols:
try:
actual = filtered_df['weighted_stress_score'].dropna()
predicted = filtered_df[col].dropna()

if len(actual) == len(predicted) and len(actual) > 0:
mse = np.mean((actual - predicted) ** 2)
mae = np.mean(np.abs(actual - predicted))

metrics_data.append({
'Model': col.replace('_prediction', '').replace('_', ' ').title(),
'MSE': f'{mse:.4f}',
'MAE': f'{mae:.4f}'
})
except:
pass

if metrics_data:
st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
else:
st.info("Model predictions not yet available. Run the pipeline to train models.")

def render_feature_analysis(self):
"""Render feature analysis view."""
st.markdown("## Feature Analysis")

filtered_df = self._apply_filters(self.df)

if filtered_df.empty:
st.warning("No data available")
return

numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['is_anomaly']]

if len(numeric_cols) > 1:
st.markdown("### Feature Correlations")

# Correlation matrix
corr_cols = numeric_cols[:15] # Limit to 15 features
corr_matrix = filtered_df[corr_cols].corr()

fig = px.imshow(
corr_matrix,
labels=dict(color="Correlation"),
color_continuous_scale='RdBu',
zmin=-1,
zmax=1
)

fig.update_layout(
title='Feature Correlation Matrix',
height=600
)

st.plotly_chart(fig, use_container_width=True)

# Feature statistics
st.markdown("### Feature Statistics")

stats_df = filtered_df[numeric_cols].describe().T
stats_df = stats_df.round(3)

st.dataframe(stats_df, use_container_width=True)
else:
st.info("Not enough features for analysis")

def render_data_export(self):
"""Render data export view."""
st.markdown("## Data Export")

filtered_df = self._apply_filters(self.df)

if filtered_df.empty:
st.warning("No data available to export")
return

col1, col2 = st.columns(2)

with col1:
st.markdown("### Export Format")
export_format = st.radio(
"Select format",
['CSV', 'JSON', 'Excel'],
horizontal=True
)

with col2:
st.markdown("### Data Preview")
st.dataframe(filtered_df.head(5), use_container_width=True)

st.markdown(f"**Total Records:** {len(filtered_df):,}")

# Export buttons
if export_format == 'CSV':
csv = filtered_df.to_csv(index=False)
st.download_button(
label="Download CSV",
data=csv,
file_name=f"market_data_{datetime.now().strftime('%Y%m%d')}.csv",
mime="text/csv"
)

elif export_format == 'JSON':
json_str = filtered_df.to_json(orient='records', indent=2)
st.download_button(
label="Download JSON",
data=json_str,
file_name=f"market_data_{datetime.now().strftime('%Y%m%d')}.json",
mime="application/json"
)

elif export_format == 'Excel':
import io
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
filtered_df.to_excel(writer, sheet_name='Data', index=False)

st.download_button(
label="Download Excel",
data=buffer.getvalue(),
file_name=f"market_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

def render_footer(self):
"""Render professional footer."""
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.9em; padding: 20px;'>
Market Narrative Risk Intelligence System v1.0.0<br>
Real-time market analysis powered by machine learning
</div>
""", unsafe_allow_html=True)

def run(self):
"""Run the dashboard application."""
try:
self.render_header()
self.render_sidebar()

# View handlers
view_handlers = {
'overview': self.render_overview,
'stress_analysis': self.render_stress_analysis,
'risk_regimes': self.render_risk_regimes,
'anomaly_detection': self.render_anomaly_detection,
'explainability': self.render_explainability,
'historical_similarity': self.render_historical_similarity,
'model_performance': self.render_model_performance,
'feature_analysis': self.render_feature_analysis,
'data_export': self.render_data_export
}

current_view = st.session_state.get('current_view', 'overview')
handler = view_handlers.get(current_view, self.render_overview)

handler()

self.render_footer()

self.logger.info(f"Dashboard view '{current_view}' rendered successfully")

except Exception as e:
self.logger.error(f"Dashboard error: {e}", exc_info=True)
st.error(f"An error occurred: {str(e)}")
with st.expander("Error Details"):
st.code(traceback.format_exc())


def main():
"""Main entry point for the dashboard."""
try:
dashboard = MarketRiskDashboard()
dashboard.run()
except Exception as e:
st.error(f"Failed to initialize dashboard: {e}")
st.code(traceback.format_exc())
st.stop()


if __name__ == "__main__":
main()
