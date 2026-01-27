"""
Market Risk Intelligence Dashboard

Professional dashboard for real-time market risk monitoring and analysis.
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
import yaml
import pickle
import logging
from typing import Dict, List, Optional
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Market Risk Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.6rem;
        color: #374151;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.875rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.25rem;
    }
    .metric-change {
        font-size: 0.875rem;
        color: #6B7280;
    }
    .alert-high {
        border-left-color: #DC2626;
    }
    .alert-medium {
        border-left-color: #F59E0B;
    }
    .alert-low {
        border-left-color: #10B981;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class MarketRiskDashboard:
    """Professional dashboard for market risk intelligence."""
    
    def __init__(self):
        """Initialize dashboard with configuration and data connections."""
        self.config = self._load_config()
        self.data_paths = self.config.get('data', {}).get('data_paths', {})
        
        # Load data
        self.market_data = self._load_market_data()
        self.features = self._load_features()
        self.models = self._load_models()
        self.monitoring_data = self._load_monitoring_data()
        
    def _load_config(self) -> Dict:
        """Load project configuration."""
        try:
            config_path = Path("configs/project_config.yaml")
            if not config_path.exists():
                config_path = Path("../configs/project_config.yaml")
            
            if not config_path.exists():
                logger.warning("Configuration file not found, using defaults")
                return {
                    'data': {
                        'data_paths': {
                            'raw': 'data/raw',
                            'processed': 'data/processed',
                            'features': 'data/features',
                            'artifacts': 'artifacts'
                        }
                    }
                }
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load sub-configs
            configs_dir = config_path.parent
            for subconfig in ['scraping', 'features', 'models', 'monitoring']:
                sub_path = configs_dir / f"{subconfig}.yaml"
                if sub_path.exists():
                    with open(sub_path, 'r') as f:
                        config[subconfig] = yaml.safe_load(f)
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {
                'data': {
                    'data_paths': {
                        'raw': 'data/raw',
                        'processed': 'data/processed',
                        'features': 'data/features',
                        'artifacts': 'artifacts'
                    }
                }
            }
    
    def _load_market_data(self) -> pd.DataFrame:
        """Load latest processed or raw market data."""
        try:
            # Try processed data first
            processed_dir = Path(self.data_paths.get('processed', 'data/processed'))
            if processed_dir.exists():
                parquet_files = list(processed_dir.glob("*.parquet"))
                csv_files = list(processed_dir.glob("*.csv"))
                
                all_files = parquet_files + csv_files
                if all_files:
                    latest_file = max(all_files, key=lambda x: x.stat().st_mtime)
                    
                    if latest_file.suffix == '.parquet':
                        df = pd.read_parquet(latest_file)
                    else:
                        df = pd.read_csv(latest_file, parse_dates=['Date'] if 'Date' in pd.read_csv(latest_file, nrows=0).columns else False)
                    
                    logger.info(f"Loaded processed data: {df.shape}")
                    return df
            
            # Fallback to raw data
            raw_dir = Path(self.data_paths.get('raw', 'data/raw'))
            if not raw_dir.exists():
                return pd.DataFrame()
            
            files = list(raw_dir.glob("*.parquet"))
            if not files:
                return pd.DataFrame()
            
            # Get latest file per ticker
            latest_files = {}
            for f in files:
                try:
                    ticker = f.stem.split('_')[0]
                    if ticker not in latest_files or f.stat().st_mtime > latest_files[ticker].stat().st_mtime:
                        latest_files[ticker] = f
                except:
                    continue
            
            # Merge data
            dfs = []
            for ticker, fpath in latest_files.items():
                df = pd.read_parquet(fpath)
                
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                else:
                    df = df.reset_index()
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                
                # Rename columns to ticker_column format
                rename_map = {}
                for col in df.columns:
                    if col != 'Date':
                        clean_col = col.replace('Adj Close', 'AdjClose').replace(' ', '_')
                        rename_map[col] = f"{ticker}_{clean_col}"
                
                df = df.rename(columns=rename_map)
                
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                
                dfs.append(df)
            
            if not dfs:
                return pd.DataFrame()
            
            market_df = pd.concat(dfs, axis=1).sort_index()
            market_df = market_df.reset_index()
            
            logger.info(f"Loaded and merged raw data: {market_df.shape}")
            return market_df
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return pd.DataFrame()
    
    def _load_features(self) -> pd.DataFrame:
        """Load latest engineered features."""
        try:
            features_dir = Path(self.data_paths.get('features', 'data/features'))
            if not features_dir.exists():
                return pd.DataFrame()
            
            feature_files = list(features_dir.glob("*features*.parquet")) + list(features_dir.glob("*features*.csv"))
            
            if not feature_files:
                return pd.DataFrame()
            
            latest_features = max(feature_files, key=lambda x: x.stat().st_mtime)
            
            if latest_features.suffix == '.parquet':
                df = pd.read_parquet(latest_features)
            else:
                df = pd.read_csv(latest_features)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            logger.info(f"Loaded features: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return pd.DataFrame()
    
    def _load_models(self) -> Dict:
        """Load trained models and metadata."""
        models = {}
        try:
            artifacts_dir = Path(self.data_paths.get('artifacts', 'artifacts')) / 'models'
            if not artifacts_dir.exists():
                return models
            
            # Load models
            for model_type in ['volatility', 'regime', 'anomaly']:
                model_files = list(artifacts_dir.glob(f"{model_type}_model_*.pkl"))
                if model_files:
                    latest = max(model_files, key=lambda x: x.stat().st_mtime)
                    with open(latest, 'rb') as f:
                        models[model_type] = pickle.load(f)
            
            # Load metrics
            metrics_dir = Path(self.data_paths.get('artifacts', 'artifacts'))
            metrics_files = list(metrics_dir.glob("*metrics*.json"))
            if metrics_files:
                latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
                with open(latest_metrics, 'r') as f:
                    models['metrics'] = json.load(f)
            
            logger.info(f"Loaded {len(models)} model components")
            return models
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return models
    
    def _load_monitoring_data(self) -> Dict:
        """Load monitoring and drift detection data."""
        monitoring_data = {}
        try:
            artifacts_dir = Path(self.data_paths.get('artifacts', 'artifacts'))
            drift_files = list(artifacts_dir.glob("*drift*.json")) + list(artifacts_dir.glob("*monitoring*.json"))
            
            for drift_file in drift_files[-3:]:
                try:
                    with open(drift_file, 'r') as f:
                        data = json.load(f)
                        report_date = drift_file.stem.split('_')[-1]
                        monitoring_data[report_date] = data
                except:
                    continue
            
            return monitoring_data
            
        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")
            return monitoring_data
    
    def render_header(self):
        """Render dashboard header."""
        st.markdown('<h1 class="main-header">Market Risk Intelligence Platform</h1>', unsafe_allow_html=True)
        st.markdown("Real-time risk monitoring and volatility analysis")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not self.market_data.empty:
                data_date = self.market_data['Date'].max() if 'Date' in self.market_data.columns else datetime.now()
                st.info(f"Data updated: {data_date.strftime('%Y-%m-%d')}")
            else:
                st.warning("No data available")
        
        with col2:
            st.info(f"Assets tracked: {len([c for c in self.market_data.columns if 'AdjClose' in str(c)])}")
        
        with col3:
            model_count = len([k for k in self.models.keys() if k != 'metrics'])
            st.info(f"Models loaded: {model_count}")
        
        with col4:
            st.info(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    
    def render_sidebar(self):
        """Render sidebar controls."""
        with st.sidebar:
            st.title("Controls")
            
            st.markdown("### Data Pipeline")
            if st.button("Run Data Pipeline", type="primary"):
                self._run_data_pipeline()
            
            if st.button("Refresh Data"):
                st.rerun()
            
            st.markdown("---")
            
            st.markdown("### Analysis Settings")
            st.date_input("Start Date", datetime.now() - timedelta(days=365))
            st.date_input("End Date", datetime.now())
            
            st.markdown("---")
            
            st.markdown("### System Information")
            if not self.market_data.empty:
                st.caption(f"Records: {len(self.market_data):,}")
                if 'Date' in self.market_data.columns:
                    date_range = f"{self.market_data['Date'].min().date()} to {self.market_data['Date'].max().date()}"
                    st.caption(f"Date range: {date_range}")
            else:
                st.caption("No data loaded")
    
    def _run_data_pipeline(self):
        """Execute data scraping pipeline."""
        try:
            from pipelines.scraping_pipeline import run_scraping_pipeline
            
            with st.spinner("Fetching market data..."):
                result = run_scraping_pipeline(self.config)
                
                if result:
                    st.success("Data pipeline completed successfully")
                    st.rerun()
                else:
                    st.error("Data pipeline failed. Check logs for details.")
                    
        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            logger.error(f"Pipeline execution failed: {e}")
    
    def render_market_overview(self):
        """Render market overview section."""
        st.markdown('<h2 class="section-header">Market Overview</h2>', unsafe_allow_html=True)
        
        if self.market_data.empty:
            st.markdown('<div class="warning-box">Market data not available. Please run the data pipeline to fetch data.</div>', unsafe_allow_html=True)
            return
        
        # Calculate key metrics
        metrics = self._calculate_market_metrics()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_metric_card("Market Return (YTD)", 
                                    metrics.get('ytd_return', 'N/A'),
                                    metrics.get('return_trend', ''))
        
        with col2:
            self._render_metric_card("Current Volatility", 
                                    metrics.get('current_volatility', 'N/A'),
                                    metrics.get('volatility_trend', ''))
        
        with col3:
            risk_level = metrics.get('risk_level', 'Medium')
            alert_class = 'alert-high' if risk_level == 'High' else 'alert-medium' if risk_level == 'Medium' else 'alert-low'
            self._render_metric_card("Risk Level", risk_level, 
                                    metrics.get('risk_change', ''), alert_class)
        
        with col4:
            self._render_metric_card("Average Correlation", 
                                    metrics.get('avg_correlation', 'N/A'),
                                    metrics.get('correlation_trend', ''))
        
        # Charts
        st.markdown("#### Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = self._create_market_performance_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = self._create_sector_performance_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        # Asset metrics table
        st.markdown("#### Asset Metrics")
        metrics_df = self._create_metrics_table()
        if not metrics_df.empty:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    def render_risk_analysis(self):
        """Render risk analysis section."""
        st.markdown('<h2 class="section-header">Risk Analysis</h2>', unsafe_allow_html=True)
        
        if self.market_data.empty:
            st.markdown('<div class="info-box">Load market data to view risk analysis</div>', unsafe_allow_html=True)
            return
        
        # Asset selector
        assets = [c.split('_')[0] for c in self.market_data.columns if 'AdjClose' in str(c)]
        
        if not assets:
            st.warning("No assets available for analysis")
            return
        
        selected_asset = st.selectbox("Select Asset for Analysis", assets)
        
        if selected_asset:
            col_name = f"{selected_asset}_AdjClose"
            stats = self._calculate_var_metrics(col_name)
            
            if stats:
                # Risk metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Value at Risk (95%)", f"{stats['var']:.2%}")
                
                with col2:
                    st.metric("Expected Shortfall (CVaR)", f"{stats['cvar']:.2%}")
                
                with col3:
                    st.metric("Maximum Drawdown", f"{stats['max_drawdown']:.2%}")
                
                # Visualization tabs
                tab1, tab2 = st.tabs(["Return Distribution", "Drawdown Analysis"])
                
                with tab1:
                    fig = self._create_return_distribution(stats)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = self._create_drawdown_chart(stats)
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_model_performance(self):
        """Render model performance section."""
        st.markdown('<h2 class="section-header">Model Performance</h2>', unsafe_allow_html=True)
        
        if not self.models:
            st.markdown('<div class="info-box">No trained models available. Run the training pipeline to enable predictive analytics.</div>', unsafe_allow_html=True)
            
            # Show expected models
            st.markdown("#### Expected Model Components")
            expected_models = pd.DataFrame({
                "Model Type": ["Volatility Forecasting", "Regime Detection", "Anomaly Detection"],
                "Purpose": ["Predict future market volatility", "Identify market regimes", "Detect unusual market behavior"],
                "Status": ["Not trained", "Not trained", "Not trained"]
            })
            st.table(expected_models)
            
            return
        
        st.success(f"Loaded {len([k for k in self.models.keys() if k != 'metrics'])} trained models")
        
        # Model information
        model_info = []
        for name, model_data in self.models.items():
            if name != 'metrics':
                model_info.append({
                    "Model Type": name.title(),
                    "Status": "Active",
                    "Last Updated": "Available"
                })
        
        if model_info:
            st.table(pd.DataFrame(model_info))
        
        # Model metrics if available
        if 'metrics' in self.models:
            st.markdown("#### Performance Metrics")
            metrics = self.models['metrics']
            
            cols = st.columns(3)
            for idx, (metric_name, metric_value) in enumerate(metrics.items()):
                with cols[idx % 3]:
                    if isinstance(metric_value, dict):
                        for k, v in metric_value.items():
                            st.metric(f"{metric_name} - {k}", f"{v:.4f}" if isinstance(v, float) else str(v))
                    else:
                        st.metric(metric_name, f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value))
    
    # Helper methods
    
    def _render_metric_card(self, title: str, value: str, change: str, alert_class: str = ""):
        """Render a metric card."""
        st.markdown(f"""
        <div class="metric-card {alert_class}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-change">{change}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def _calculate_market_metrics(self) -> Dict:
        """Calculate key market metrics."""
        metrics = {}
        
        if self.market_data.empty or 'Date' not in self.market_data.columns:
            return metrics
        
        try:
            price_cols = [col for col in self.market_data.columns if 'AdjClose' in str(col)]
            if not price_cols:
                return metrics
            
            # Use SPY or first available ticker
            target_col = next((c for c in price_cols if 'SPY' in c), price_cols[0])
            
            # YTD return
            prices = self.market_data[['Date', target_col]].dropna()
            prices['Date'] = pd.to_datetime(prices['Date'])
            current_year = datetime.now().year
            ytd_mask = prices['Date'].dt.year == current_year
            
            if ytd_mask.any():
                ytd_prices = prices[ytd_mask]
                if len(ytd_prices) > 1:
                    ytd_return = (ytd_prices.iloc[-1][target_col] / ytd_prices.iloc[0][target_col]) - 1
                    metrics['ytd_return'] = f"{ytd_return:.2%}"
                    metrics['return_trend'] = "Positive" if ytd_return > 0 else "Negative"
            
            # Volatility
            returns = np.log(prices[target_col] / prices[target_col].shift(1)).dropna()
            if len(returns) >= 20:
                current_vol = returns.tail(20).std() * np.sqrt(252)
                metrics['current_volatility'] = f"{current_vol:.2%}"
                
                # Risk level
                if current_vol < 0.15:
                    metrics['risk_level'] = 'Low'
                elif current_vol < 0.25:
                    metrics['risk_level'] = 'Medium'
                else:
                    metrics['risk_level'] = 'High'
            
            # Correlation
            if len(price_cols) >= 3:
                price_data = self.market_data[price_cols[:5]].dropna()
                returns_data = np.log(price_data / price_data.shift(1)).dropna()
                if not returns_data.empty:
                    corr_matrix = returns_data.corr()
                    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean()
                    metrics['avg_correlation'] = f"{avg_corr:.3f}"
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return metrics
    
    def _calculate_var_metrics(self, ticker_col: str) -> Optional[Dict]:
        """Calculate Value at Risk metrics."""
        if ticker_col not in self.market_data.columns:
            return None
        
        try:
            prices = self.market_data[['Date', ticker_col]].dropna()
            prices['Date'] = pd.to_datetime(prices['Date'])
            prices = prices.set_index('Date')[ticker_col]
            
            returns = prices.pct_change().dropna()
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'var': var_95,
                'cvar': cvar_95,
                'max_drawdown': max_drawdown,
                'returns': returns,
                'prices': prices,
                'drawdown': drawdown
            }
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return None
    
    def _create_market_performance_chart(self):
        """Create market performance chart."""
        fig = go.Figure()
        
        if self.market_data.empty or 'Date' not in self.market_data.columns:
            return fig
        
        try:
            price_cols = [col for col in self.market_data.columns if 'AdjClose' in str(col)][:3]
            
            for col in price_cols:
                prices = self.market_data[['Date', col]].dropna()
                if len(prices) > 0:
                    ticker = str(col).split('_')[0]
                    normalized = (prices[col] / prices[col].iloc[0]) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=prices['Date'],
                        y=normalized,
                        name=ticker,
                        mode='lines',
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title="Normalized Performance",
                xaxis_title="Date",
                yaxis_title="Normalized Price (Base=100)",
                hovermode='x unified',
                height=400,
                template="plotly_white"
            )
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
        
        return fig
    
    def _create_sector_performance_chart(self):
        """Create sector performance chart."""
        fig = go.Figure()
        
        try:
            sectors = ['XLF', 'XLK', 'XLV', 'XLE', 'XLY', 'XLI']
            sector_returns = []
            sector_names = []
            
            for sector in sectors:
                col = f"{sector}_AdjClose"
                if col in self.market_data.columns:
                    prices = self.market_data[['Date', col]].dropna()
                    if len(prices) >= 20:
                        recent_return = (prices[col].iloc[-1] / prices[col].iloc[-20] - 1) * 100
                        sector_names.append(sector)
                        sector_returns.append(recent_return)
            
            if sector_returns:
                colors = ['#DC2626' if x < 0 else '#10B981' for x in sector_returns]
                
                fig.add_trace(go.Bar(
                    x=sector_names,
                    y=sector_returns,
                    marker_color=colors,
                    text=[f"{x:.1f}%" for x in sector_returns],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Sector Performance (20-Day)",
                    xaxis_title="Sector",
                    yaxis_title="Return (%)",
                    height=400,
                    template="plotly_white"
                )
        
        except Exception as e:
            logger.error(f"Error creating sector chart: {e}")
        
        return fig
    
    def _create_return_distribution(self, stats: Dict):
        """Create return distribution chart."""
        fig = go.Figure()
        
        returns = stats['returns']
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker_color='#3B82F6',
            opacity=0.7
        ))
        
        # Add VaR line
        fig.add_vline(
            x=stats['var'],
            line_dash="dash",
            line_color="#DC2626",
            annotation_text=f"VaR 95%: {stats['var']:.2%}"
        )
        
        fig.update_layout(
            title="Return Distribution",
            xaxis_title="Returns",
            yaxis_title="Frequency",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def _create_drawdown_chart(self, stats: Dict):
        """Create drawdown chart."""
        fig = go.Figure()
        
        drawdown = stats['drawdown']
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.3)',
            line=dict(color='#DC2626', width=2),
            name='Drawdown'
        ))
        
        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def _create_metrics_table(self):
        """Create asset metrics table."""
        try:
            if self.market_data.empty:
                return pd.DataFrame()
            
            metrics = []
            price_cols = [col for col in self.market_data.columns if 'AdjClose' in str(col)][:8]
            
            for col in price_cols:
                ticker = str(col).split('_')[0]
                prices = self.market_data[['Date', col]].dropna()
                
                if len(prices) >= 20:
                    returns = np.log(prices[col] / prices[col].shift(1)).dropna()
                    ret_20d = (prices[col].iloc[-1] / prices[col].iloc[-20] - 1) * 100
                    volatility = returns.tail(20).std() * np.sqrt(252) * 100
                    
                    metrics.append({
                        'Ticker': ticker,
                        'Current Price': f"${prices[col].iloc[-1]:.2f}",
                        '20-Day Return': f"{ret_20d:+.2f}%",
                        'Volatility': f"{volatility:.2f}%"
                    })
            
            return pd.DataFrame(metrics)
            
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            return pd.DataFrame()
    
    def run_dashboard(self):
        """Main dashboard execution."""
        try:
            # Render sidebar
            self.render_sidebar()
            
            # Render header
            self.render_header()
            
            if self.market_data.empty:
                st.markdown('<div class="warning-box">No market data available. Click "Run Data Pipeline" in the sidebar to fetch data.</div>', unsafe_allow_html=True)
                return
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs([
                "Market Overview",
                "Risk Analysis",
                "Model Performance"
            ])
            
            with tab1:
                self.render_market_overview()
            
            with tab2:
                self.render_risk_analysis()
            
            with tab3:
                self.render_model_performance()
            
            # Footer
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption(f"Data records: {len(self.market_data):,}")
                st.caption(f"Features: {len(self.features.columns) if not self.features.empty else 0}")
            
            with col2:
                st.caption(f"Models loaded: {len([k for k in self.models.keys() if k != 'metrics'])}")
                st.caption("Market Risk Intelligence Platform")
            
        except Exception as e:
            logger.error(f"Dashboard error: {e}", exc_info=True)
            st.error(f"Error: {str(e)}")


def main():
    """Main entry point."""
    try:
        dashboard = MarketRiskDashboard()
        dashboard.run_dashboard()
    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        st.error(f"Failed to initialize dashboard: {str(e)}")


if __name__ == "__main__":
    main()
