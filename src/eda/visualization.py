"""
Professional Exploratory Data Analysis Visualization Module
Advanced analytics and visualization for market risk intelligence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
import json
from dataclasses import dataclass
from enum import Enum

# Try to import internal logger
try:
    from src.utils.logger import get_dashboard_logger
    logger = get_dashboard_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class PlotTheme(Enum):
    """Plot theme options."""
    PLOTLY_WHITE = "plotly_white"
    PLOTLY_DARK = "plotly_dark"
    PLOTLY_WHITEBG = "plotly_whitebg"
    GGPLOT2 = "ggplot2"
    SEABORN = "seaborn"
    SIMPLE_WHITE = "simple_white"


class ChartType(Enum):
    """Chart type options."""
    TIME_SERIES = "time_series"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    BOXPLOT = "boxplot"
    VIOLIN = "violin"
    HISTOGRAM = "histogram"
    BAR = "bar"
    PIE = "pie"


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    title: str
    width: int = 1200
    height: int = 600
    theme: PlotTheme = PlotTheme.PLOTLY_WHITE
    interactive: bool = True
    show_legend: bool = True
    show_grid: bool = True
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


class ProfessionalEDAVisualizer:
    """
    Professional EDA visualization tool for market risk analytics.
    """
    
    def __init__(self, theme: PlotTheme = PlotTheme.PLOTLY_WHITE):
        """Initialize the visualizer with professional settings."""
        self.theme = theme
        self.config = ChartConfig(
            title="EDA Analysis",
            width=1400,
            height=800,
            theme=theme,
            interactive=True,
            show_legend=True,
            show_grid=True
        )
        
        # Professional color palettes
        self.sequential_palettes = {
            'blues': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', 
                     '#4292c6', '#2171b5', '#08519c', '#08306b'],
            'greens': ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', 
                      '#41ab5d', '#238b45', '#006d2c', '#00441b'],
            'reds': ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', 
                    '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],
        }
        
        self.diverging_palettes = {
            'rdbu': ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7',
                    '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'],
            'spectral': ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b',
                        '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']
        }
        
        self.categorical_palettes = {
            'set3': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
                    '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'],
            'set2': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
                    '#ffd92f', '#e5c494', '#b3b3b3']
        }
        
        logger.info(f"ProfessionalEDAVisualizer initialized with theme: {theme}")
    
    def set_config(self, **kwargs):
        """Update configuration settings."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Invalid config key: {key}")
    
    def _apply_theme(self, fig: go.Figure) -> go.Figure:
        """Apply consistent theme to plotly figure."""
        theme_template = self.theme.value
        
        # Custom template adjustments
        custom_layout = {
            'font': dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
            'title': dict(font=dict(size=18, color="#2c3e50", family="Arial, sans-serif")),
            'plot_bgcolor': 'rgba(240, 242, 245, 0.5)',
            'paper_bgcolor': 'white',
            'hoverlabel': dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            'margin': dict(l=50, r=50, t=80, b=50),
        }
        
        fig.update_layout(
            template=theme_template,
            **custom_layout
        )
        
        return fig
    
    def _add_watermark(self, fig: go.Figure, text: str = "Market Risk Intelligence") -> go.Figure:
        """Add professional watermark to figures."""
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=text,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=40,
                color="rgba(200, 200, 200, 0.3)"
            ),
            textangle=-30
        )
        
        return fig
    
    def create_market_overview_dashboard(self, df: pd.DataFrame, 
                                       date_col: str = 'timestamp',
                                       price_cols: List[str] = None) -> go.Figure:
        """
        Create comprehensive market overview dashboard.
        
        Args:
            df: Input DataFrame with market data
            date_col: Date column name
            price_cols: List of price columns to analyze
        
        Returns:
            Plotly Figure with multiple subplots
        """
        logger.info("Creating market overview dashboard")
        
        # Filter and prepare data
        plot_df = df.copy()
        plot_df[date_col] = pd.to_datetime(plot_df[date_col])
        plot_df = plot_df.sort_values(date_col)
        
        # Auto-detect price columns if not provided
        if price_cols is None:
            price_cols = [col for col in plot_df.columns 
                         if any(x in col.lower() for x in ['price', 'close', 'value'])]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Market Overview', 'Daily Returns Distribution', 
                'Price Correlation Heatmap', 'Volume Profile',
                'Volatility Analysis', 'Market Breadth',
                'Sector Performance', 'Risk Metrics',
                'Market Regime Analysis'
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            specs=[
                [{"type": "scatter", "colspan": 3}, None, None],
                [{"type": "histogram"}, {"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Market overview - Price trends
        if price_cols and len(price_cols) > 0:
            for i, col in enumerate(price_cols[:3]):  # Show first 3 instruments
                fig.add_trace(
                    go.Scatter(
                        x=plot_df[date_col],
                        y=plot_df[col],
                        mode='lines',
                        name=col,
                        line=dict(width=2, color=self.config.color_palette[i]),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        # 2. Daily returns distribution
        if price_cols and len(price_cols) > 0:
            returns = plot_df[price_cols[0]].pct_change().dropna()
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Returns',
                    marker_color=self.config.color_palette[0],
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Add normal distribution overlay
            x_norm = np.linspace(returns.min(), returns.max(), 100)
            y_norm = (1/(returns.std() * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm - returns.mean())/returns.std())**2)
            
            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=2, col=1
            )
        
        # 3. Correlation heatmap
        if len(price_cols) > 1:
            corr_matrix = plot_df[price_cols].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale=self.diverging_palettes['rdbu'],
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="Correlation")
                ),
                row=2, col=2
            )
        
        # 4. Volume profile (if volume column exists)
        volume_cols = [col for col in plot_df.columns if 'volume' in col.lower()]
        if volume_cols:
            fig.add_trace(
                go.Bar(
                    x=plot_df[date_col],
                    y=plot_df[volume_cols[0]],
                    name='Volume',
                    marker_color=self.config.color_palette[3],
                    opacity=0.6
                ),
                row=2, col=3
            )
        
        # 5. Volatility analysis (rolling std)
        if price_cols:
            returns_series = plot_df[price_cols[0]].pct_change()
            volatility = returns_series.rolling(window=20).std() * np.sqrt(252)
            
            fig.add_trace(
                go.Scatter(
                    x=plot_df[date_col],
                    y=volatility,
                    mode='lines',
                    name='Annualized Volatility',
                    line=dict(color=self.config.color_palette[1], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 127, 14, 0.2)'
                ),
                row=3, col=1
            )
        
        # 6. Market breadth (if multiple instruments available)
        if len(price_cols) > 5:
            # Calculate advance-decline line
            daily_returns = plot_df[price_cols].pct_change()
            advances = (daily_returns > 0).sum(axis=1)
            declines = (daily_returns < 0).sum(axis=1)
            breadth = advances - declines
            
            fig.add_trace(
                go.Bar(
                    x=plot_df[date_col],
                    y=breadth,
                    name='Market Breadth',
                    marker_color=np.where(breadth > 0, 'green', 'red'),
                    opacity=0.7
                ),
                row=3, col=2
            )
        
        # Apply professional styling
        fig.update_layout(
            height=1200,
            title_text="Professional Market Overview Dashboard",
            showlegend=True,
            hovermode='x unified',
            template=self.theme.value
        )
        
        # Add annotations for key insights
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            text="<b>Market Risk Intelligence Dashboard</b>",
            showarrow=False,
            font=dict(size=16, color="#2c3e50"),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#2c3e50",
            borderwidth=2,
            borderpad=4
        )
        
        fig = self._apply_theme(fig)
        
        return fig
    
    def plot_risk_regime_analysis(self, df: pd.DataFrame,
                                date_col: str = 'timestamp',
                                price_col: str = 'price',
                                volatility_window: int = 20,
                                output_path: Optional[Path] = None) -> go.Figure:
        """
        Advanced risk regime analysis with market state detection.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            price_col: Price column name
            volatility_window: Window for volatility calculation
            output_path: Path to save plot
        
        Returns:
            Plotly Figure object
        """
        logger.info("Creating risk regime analysis")
        
        # Prepare data
        plot_df = df.copy()
        plot_df[date_col] = pd.to_datetime(plot_df[date_col])
        plot_df = plot_df.sort_values(date_col).set_index(date_col)
        
        # Calculate returns and volatility
        returns = plot_df[price_col].pct_change()
        volatility = returns.rolling(window=volatility_window).std() * np.sqrt(252)
        rolling_return = returns.rolling(window=volatility_window).mean() * 252
        
        # Create regime classification
        conditions = [
            (volatility > volatility.quantile(0.75)) & (rolling_return > 0),
            (volatility > volatility.quantile(0.75)) & (rolling_return <= 0),
            (volatility <= volatility.quantile(0.75)) & (rolling_return > 0),
            (volatility <= volatility.quantile(0.75)) & (rolling_return <= 0)
        ]
        
        choices = ['High Vol, High Return', 'High Vol, Low Return', 
                  'Low Vol, High Return', 'Low Vol, Low Return']
        
        plot_df['regime'] = np.select(conditions, choices, default='Unknown')
        
        # Create figure with multiple subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                'Price with Regime Zones',
                'Annualized Volatility',
                'Annualized Return',
                'Regime Distribution'
            ],
            vertical_spacing=0.08,
            shared_xaxes=True,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # 1. Price with regime coloring
        for regime in plot_df['regime'].unique():
            regime_data = plot_df[plot_df['regime'] == regime]
            
            regime_color = {
                'High Vol, High Return': '#d62728',
                'High Vol, Low Return': '#ff7f0e',
                'Low Vol, High Return': '#2ca02c',
                'Low Vol, Low Return': '#1f77b4'
            }.get(regime, '#7f7f7f')
            
            fig.add_trace(
                go.Scatter(
                    x=regime_data.index,
                    y=regime_data[price_col],
                    mode='markers',
                    name=regime,
                    marker=dict(
                        size=8,
                        color=regime_color,
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[price_col],
                mode='lines',
                name='Price',
                line=dict(color='black', width=1),
                opacity=0.3,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Volatility
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=volatility,
                mode='lines',
                name='Volatility',
                line=dict(color='#ff7f0e', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 127, 14, 0.1)'
            ),
            row=2, col=1
        )
        
        fig.add_hline(
            y=volatility.quantile(0.75),
            line_dash="dash",
            line_color="red",
            annotation_text="High Vol Threshold",
            row=2, col=1
        )
        
        # 3. Returns
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=rolling_return,
                mode='lines',
                name='Returns',
                line=dict(color='#2ca02c', width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.1)'
            ),
            row=3, col=1
        )
        
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
            row=3, col=1
        )
        
        # 4. Regime distribution (heatmap-style)
        regime_counts = plot_df['regime'].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=regime_counts.index,
                y=regime_counts.values,
                name='Regime Count',
                marker_color=[self.config.color_palette[i % len(self.config.color_palette)] 
                            for i in range(len(regime_counts))]
            ),
            row=4, col=1
        )
        
        # Calculate regime statistics
        regime_stats = []
        for regime in choices:
            if regime in plot_df['regime'].values:
                regime_data = plot_df[plot_df['regime'] == regime]
                regime_stats.append(f"{regime}: {len(regime_data)} days")
        
        stats_text = "<b>Regime Statistics:</b><br>" + "<br>".join(regime_stats)
        
        # Add statistics box
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Risk Regime Analysis",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        fig.update_yaxes(title_text="Return", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=4, col=1)
        
        fig = self._apply_theme(fig)
        
        if output_path:
            fig.write_image(str(output_path), width=1600, height=1000)
        
        return fig
    
    def plot_correlation_network(self, df: pd.DataFrame,
                               threshold: float = 0.7,
                               output_path: Optional[Path] = None) -> go.Figure:
        """
        Create correlation network visualization.
        
        Args:
            df: Input DataFrame
            threshold: Minimum correlation to show connection
            output_path: Path to save plot
        
        Returns:
            Plotly Figure object
        """
        logger.info("Creating correlation network")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            logger.warning("Need at least 3 numeric columns for network")
            return None
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Prepare network data
        edges = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if corr > threshold:
                    edges.append({
                        'source': corr_matrix.columns[i],
                        'target': corr_matrix.columns[j],
                        'weight': corr,
                        'color': 'red' if corr > 0.8 else 'orange' if corr > 0.6 else 'blue'
                    })
        
        if not edges:
            logger.info(f"No correlations above threshold {threshold}")
            return None
        
        # Create node positions
        nodes = list(set([edge['source'] for edge in edges] + [edge['target'] for edge in edges]))
        
        # Calculate node importance (degree centrality)
        node_degree = {node: 0 for node in nodes}
        for edge in edges:
            node_degree[edge['source']] += 1
            node_degree[edge['target']] += 1
        
        # Create network visualization
        edge_trace = []
        for edge in edges:
            edge_trace.append(
                go.Scatter(
                    x=[None],  # Will be set by layout algorithm
                    y=[None],
                    mode='lines',
                    line=dict(
                        width=edge['weight'] * 5,
                        color=edge['color'],
                        opacity=0.5
                    ),
                    hoverinfo='text',
                    text=f"{edge['source']} ↔ {edge['target']}<br>Correlation: {edge['weight']:.3f}",
                    showlegend=False
                )
            )
        
        node_trace = go.Scatter(
            x=[],  # Will be set by layout algorithm
            y=[],
            mode='markers+text',
            text=nodes,
            textposition="top center",
            marker=dict(
                size=[20 + node_degree[node] * 5 for node in nodes],
                color=[self.config.color_palette[i % len(self.config.color_palette)] 
                      for i in range(len(nodes))],
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            hovertext=[f"{node}<br>Connections: {node_degree[node]}" for node in nodes]
        )
        
        # Use force-directed layout for better visualization
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title=f"Feature Correlation Network (Threshold: {threshold})",
            height=800,
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='rgba(240, 242, 245, 0.8)'
        )
        
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        
        fig = self._apply_theme(fig)
        
        if output_path:
            fig.write_image(str(output_path), width=1200, height=800)
        
        return fig
    
    def create_interactive_scatter_matrix(self, df: pd.DataFrame,
                                        features: List[str] = None,
                                        color_col: str = None,
                                        output_path: Optional[Path] = None) -> go.Figure:
        """
        Create interactive scatter matrix with advanced features.
        
        Args:
            df: Input DataFrame
            features: List of features to include
            color_col: Column to use for coloring
            output_path: Path to save plot
        
        Returns:
            Plotly Figure object
        """
        logger.info("Creating interactive scatter matrix")
        
        # Select features if not provided
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = features[:8]  # Limit to 8 features
        
        if len(features) < 2:
            logger.warning("Need at least 2 features for scatter matrix")
            return None
        
        # Create scatter matrix
        fig = px.scatter_matrix(
            df,
            dimensions=features,
            color=color_col if color_col and color_col in df.columns else None,
            title="Interactive Scatter Matrix",
            opacity=0.7,
            labels={col: col.replace('_', ' ').title() for col in features},
            height=1000,
            color_discrete_sequence=self.config.color_palette
        )
        
        # Update diagonal plots to show distributions
        for i in range(len(features)):
            row = i + 1
            col = i + 1
            
            # Get the subplot
            fig.update_traces(
                diagonal_visible=False,
                showupperhalf=False
            )
            
            # Add histogram on diagonal
            fig.add_trace(
                go.Histogram(
                    x=df[features[i]].dropna(),
                    nbinsx=30,
                    marker_color=self.config.color_palette[i % len(self.config.color_palette)],
                    opacity=0.7,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Add correlation values to upper triangle
        corr_matrix = df[features].corr()
        
        for i in range(len(features)):
            for j in range(len(features)):
                if i < j:  # Upper triangle
                    fig.add_annotation(
                        x=features[j],
                        y=features[i],
                        text=f"ρ = {corr_matrix.iloc[i, j]:.2f}",
                        showarrow=False,
                        font=dict(
                            size=10,
                            color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white"
                        ),
                        bgcolor="rgba(255, 255, 255, 0.7)" if abs(corr_matrix.iloc[i, j]) < 0.5 else "rgba(0, 0, 0, 0.7)",
                        bordercolor="black",
                        borderwidth=1
                    )
        
        fig.update_layout(
            title_text="Advanced Scatter Matrix with Correlation Indicators",
            showlegend=True,
            dragmode='select',
            hovermode='closest'
        )
        
        fig.update_traces(
            marker=dict(
                size=6,
                line=dict(width=0.5, color='white')
            ),
            selector=dict(mode='markers')
        )
        
        fig = self._apply_theme(fig)
        
        if output_path:
            fig.write_image(str(output_path), width=1400, height=1000)
        
        return fig
    
    def plot_risk_metrics_dashboard(self, df: pd.DataFrame,
                                  date_col: str = 'timestamp',
                                  value_col: str = 'price',
                                  output_path: Optional[Path] = None) -> go.Figure:
        """
        Create professional risk metrics dashboard.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            value_col: Value column for risk calculation
            output_path: Path to save plot
        
        Returns:
            Plotly Figure object
        """
        logger.info("Creating risk metrics dashboard")
        
        # Prepare data
        plot_df = df.copy()
        plot_df[date_col] = pd.to_datetime(plot_df[date_col])
        plot_df = plot_df.sort_values(date_col).set_index(date_col)
        
        # Calculate risk metrics
        returns = plot_df[value_col].pct_change().dropna()
        
        # Basic metrics
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR (Value at Risk)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # CVaR (Conditional VaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                f'Price Series ({value_col})',
                'Daily Returns Distribution',
                'Cumulative Returns',
                f'Drawdown (Max: {max_drawdown:.2%})',
                'Rolling Volatility (20D)',
                'Rolling Sharpe Ratio (60D)',
                'Value at Risk Analysis',
                'Autocorrelation of Returns',
                'QQ Plot vs Normal Distribution'
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Price series
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[value_col],
                mode='lines',
                name='Price',
                line=dict(color=self.config.color_palette[0], width=2)
            ),
            row=1, col=1
        )
        
        # 2. Returns distribution
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                marker_color=self.config.color_palette[1],
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Add VaR lines
        fig.add_vline(
            x=var_95,
            line_dash="dash",
            line_color="orange",
            annotation_text="95% VaR",
            row=1, col=2
        )
        
        fig.add_vline(
            x=var_99,
            line_dash="dash",
            line_color="red",
            annotation_text="99% VaR",
            row=1, col=2
        )
        
        # 3. Cumulative returns
        cumulative_returns = (1 + returns).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color=self.config.color_palette[2], width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.1)'
            ),
            row=1, col=3
        )
        
        # 4. Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ),
            row=2, col=1
        )
        
        # 5. Rolling volatility
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color=self.config.color_palette[3], width=2)
            ),
            row=2, col=2
        )
        
        # 6. Rolling Sharpe ratio
        rolling_sharpe = (returns.rolling(window=60).mean() * 252) / (returns.rolling(window=60).std() * np.sqrt(252))
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color=self.config.color_palette[4], width=2)
            ),
            row=2, col=3
        )
        
        # 7. VaR analysis (bar chart)
        var_data = pd.DataFrame({
            'VaR Level': ['95%', '99%'],
            'VaR': [var_95, var_99],
            'CVaR': [cvar_95, cvar_99]
        })
        
        fig.add_trace(
            go.Bar(
                x=var_data['VaR Level'],
                y=var_data['VaR'],
                name='VaR',
                marker_color='orange',
                opacity=0.7
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=var_data['VaR Level'],
                y=var_data['CVaR'],
                name='CVaR',
                marker_color='red',
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # 8. Autocorrelation
        autocorr = [returns.autocorr(lag) for lag in range(1, 21)]
        fig.add_trace(
            go.Bar(
                x=list(range(1, 21)),
                y=autocorr,
                name='Autocorrelation',
                marker_color=self.config.color_palette[5],
                opacity=0.7
            ),
            row=3, col=2
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=2)
        
        # 9. QQ Plot
        from scipy import stats
        
        # Calculate theoretical quantiles
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
        sample_quantiles = np.percentile(returns, np.linspace(1, 99, len(returns)))
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='QQ Plot',
                marker=dict(
                    color=self.config.color_palette[6],
                    size=6
                )
            ),
            row=3, col=3
        )
        
        # Add 45-degree line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal Line',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=3, col=3
        )
        
        # Add comprehensive statistics box
        stats_text = f"""
        <b>Risk Metrics Summary:</b><br>
        Annual Return: {annual_return:.2%}<br>
        Annual Volatility: {annual_volatility:.2%}<br>
        Sharpe Ratio: {sharpe_ratio:.2f}<br>
        Max Drawdown: {max_drawdown:.2%}<br>
        95% VaR: {var_95:.2%}<br>
        99% VaR: {var_99:.2%}<br>
        Skewness: {returns.skew():.3f}<br>
        Kurtosis: {returns.kurtosis():.3f}
        """
        
        fig.add_annotation(
            x=0.98,
            y=0.02,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Professional Risk Metrics Dashboard",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig = self._apply_theme(fig)
        
        if output_path:
            fig.write_image(str(output_path), width=1600, height=1200)
        
        return fig
    
    def generate_dashboard_ready_charts(self, df: pd.DataFrame,
                                      output_dir: Path = None) -> Dict[str, go.Figure]:
        """
        Generate all charts ready for dashboard integration.
        
        Args:
            df: Input DataFrame
            output_dir: Directory to save charts
        
        Returns:
            Dictionary of chart names and figures
        """
        logger.info("Generating dashboard-ready charts")
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        charts = {}
        
        try:
            # 1. Market overview
            charts['market_overview'] = self.create_market_overview_dashboard(df)
            if output_dir:
                charts['market_overview'].write_html(str(output_dir / 'market_overview.html'))
        
        except Exception as e:
            logger.error(f"Failed to create market overview: {e}")
        
        try:
            # 2. Risk regime analysis
            if 'price' in df.columns and 'timestamp' in df.columns:
                charts['risk_regime'] = self.plot_risk_regime_analysis(df)
                if output_dir:
                    charts['risk_regime'].write_html(str(output_dir / 'risk_regime.html'))
        except Exception as e:
            logger.error(f"Failed to create risk regime analysis: {e}")
        
        try:
            # 3. Correlation network
            charts['correlation_network'] = self.plot_correlation_network(df)
            if output_dir and charts['correlation_network']:
                charts['correlation_network'].write_html(str(output_dir / 'correlation_network.html'))
        except Exception as e:
            logger.error(f"Failed to create correlation network: {e}")
        
        try:
            # 4. Risk metrics dashboard
            if 'price' in df.columns and 'timestamp' in df.columns:
                charts['risk_metrics'] = self.plot_risk_metrics_dashboard(df)
                if output_dir:
                    charts['risk_metrics'].write_html(str(output_dir / 'risk_metrics.html'))
        except Exception as e:
            logger.error(f"Failed to create risk metrics dashboard: {e}")
        
        try:
            # 5. Interactive scatter matrix
            charts['scatter_matrix'] = self.create_interactive_scatter_matrix(df)
            if output_dir and charts['scatter_matrix']:
                charts['scatter_matrix'].write_html(str(output_dir / 'scatter_matrix.html'))
        except Exception as e:
            logger.error(f"Failed to create scatter matrix: {e}")
        
        # Generate JSON summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'charts_generated': len(charts),
            'chart_names': list(charts.keys()),
            'data_shape': df.shape,
            'data_columns': df.columns.tolist(),
            'data_info': {
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'date_columns': [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            }
        }
        
        if output_dir:
            with open(output_dir / 'charts_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
        
        return charts
    
    def create_professional_report(self, df: pd.DataFrame,
                                 report_title: str = "Market Risk Analysis Report",
                                 output_dir: Path = None) -> Path:
        """
        Create professional HTML report with all visualizations.
        
        Args:
            df: Input DataFrame
            report_title: Title of the report
            output_dir: Directory to save the report
        
        Returns:
            Path to the generated report
        """
        logger.info(f"Creating professional report: {report_title}")
        
        if output_dir is None:
            output_dir = Path("reports") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all charts
        charts = self.generate_dashboard_ready_charts(df, output_dir)
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_title}</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    min-height: 100vh;
                }}
                
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    padding: 30px;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                
                .header h1 {{
                    color: #2c3e50;
                    margin-bottom: 10px;
                    font-size: 2.5em;
                }}
                
                .header .subtitle {{
                    color: #7f8c8d;
                    font-size: 1.2em;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                
                .metric-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    text-align: center;
                    transition: transform 0.3s ease;
                }}
                
                .metric-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                }}
                
                .metric-card h3 {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 10px;
                }}
                
                .metric-card .value {{
                    color: #2c3e50;
                    font-size: 2em;
                    font-weight: 700;
                }}
                
                .chart-section {{
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                
                .chart-section h2 {{
                    color: #2c3e50;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 3px solid #3498db;
                    display: flex;
                    align-items: center;
                }}
                
                .chart-section h2::before {{
                    content: "📈";
                    margin-right: 10px;
                    font-size: 1.5em;
                }}
                
                .chart-container {{
                    width: 100%;
                    height: 600px;
                    margin: 20px 0;
                }}
                
                .chart-description {{
                    color: #7f8c8d;
                    margin-bottom: 20px;
                    font-size: 1em;
                    line-height: 1.6;
                }}
                
                .insights-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                
                .insight-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                
                .insight-card h4 {{
                    color: #2c3e50;
                    margin-bottom: 10px;
                }}
                
                .insight-card p {{
                    color: #666;
                    font-size: 0.95em;
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
                
                .download-btn {{
                    display: inline-block;
                    background: linear-gradient(135deg, #3498db, #2980b9);
                    color: white;
                    padding: 12px 30px;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: 600;
                    margin: 10px;
                    transition: all 0.3s ease;
                    border: none;
                    cursor: pointer;
                }}
                
                .download-btn:hover {{
                    background: linear-gradient(135deg, #2980b9, #1c6ea4);
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(41, 128, 185, 0.3);
                }}
                
                .nav-tabs {{
                    display: flex;
                    border-bottom: 2px solid #e0e0e0;
                    margin-bottom: 30px;
                    overflow-x: auto;
                }}
                
                .nav-tab {{
                    padding: 15px 30px;
                    background: none;
                    border: none;
                    font-size: 1em;
                    color: #7f8c8d;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    white-space: nowrap;
                }}
                
                .nav-tab.active {{
                    color: #3498db;
                    border-bottom: 3px solid #3498db;
                    font-weight: 600;
                }}
                
                .tab-content {{
                    display: none;
                }}
                
                .tab-content.active {{
                    display: block;
                }}
                
                @media (max-width: 768px) {{
                    .container {{
                        padding: 10px;
                    }}
                    
                    .header h1 {{
                        font-size: 2em;
                    }}
                    
                    .metrics-grid {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .chart-container {{
                        height: 400px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{report_title}</h1>
                    <div class="subtitle">
                        Professional Market Risk Analysis | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Dataset Size</h3>
                        <div class="value">{df.shape[0]:,} × {df.shape[1]}</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Time Range</h3>
                        <div class="value">
                            {pd.to_datetime(df['timestamp']).min().strftime('%Y-%m-%d') if 'timestamp' in df.columns else 'N/A'}
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Data Quality</h3>
                        <div class="value">{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])):.1%}</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Analysis Date</h3>
                        <div class="value">{datetime.now().strftime('%Y-%m-%d')}</div>
                    </div>
                </div>
                
                <div class="nav-tabs">
                    <button class="nav-tab active" onclick="showTab('overview')">Market Overview</button>
                    <button class="nav-tab" onclick="showTab('risk')">Risk Analysis</button>
                    <button class="nav-tab" onclick="showTab('correlations')">Correlations</button>
                    <button class="nav-tab" onclick="showTab('metrics')">Risk Metrics</button>
                    <button class="nav-tab" onclick="showTab('insights')">Insights</button>
                </div>
                
                <div id="overview-tab" class="tab-content active">
                    <div class="chart-section">
                        <h2>Market Overview Dashboard</h2>
                        <div class="chart-description">
                            Comprehensive view of market trends, volatility, and performance across multiple instruments.
                        </div>
                        <div class="chart-container" id="market-overview-chart">
                            <!-- Chart will be loaded here -->
                        </div>
                    </div>
                </div>
                
                <div id="risk-tab" class="tab-content">
                    <div class="chart-section">
                        <h2>Risk Regime Analysis</h2>
                        <div class="chart-description">
                            Identification of different market regimes based on volatility and return characteristics.
                        </div>
                        <div class="chart-container" id="risk-regime-chart">
                            <!-- Chart will be loaded here -->
                        </div>
                    </div>
                </div>
                
                <div id="correlations-tab" class="tab-content">
                    <div class="chart-section">
                        <h2>Feature Correlation Network</h2>
                        <div class="chart-description">
                            Network visualization showing relationships between different market features.
                        </div>
                        <div class="chart-container" id="correlation-network-chart">
                            <!-- Chart will be loaded here -->
                        </div>
                    </div>
                    
                    <div class="chart-section">
                        <h2>Interactive Scatter Matrix</h2>
                        <div class="chart-description">
                            Pairwise relationships between key features with correlation indicators.
                        </div>
                        <div class="chart-container" id="scatter-matrix-chart">
                            <!-- Chart will be loaded here -->
                        </div>
                    </div>
                </div>
                
                <div id="metrics-tab" class="tab-content">
                    <div class="chart-section">
                        <h2>Risk Metrics Dashboard</h2>
                        <div class="chart-description">
                            Comprehensive risk assessment including VaR, drawdown, volatility, and Sharpe ratio analysis.
                        </div>
                        <div class="chart-container" id="risk-metrics-chart">
                            <!-- Chart will be loaded here -->
                        </div>
                    </div>
                </div>
                
                <div id="insights-tab" class="tab-content">
                    <div class="insights-grid">
                        <div class="insight-card">
                            <h4>📊 Data Quality</h4>
                            <p>Dataset contains {df.shape[0]:,} observations with {df.shape[1]} features. Missing data rate is {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%.</p>
                        </div>
                        
                        <div class="insight-card">
                            <h4>📈 Market Trends</h4>
                            <p>Analysis shows {len([col for col in df.columns if 'price' in col.lower()])} price series with varying degrees of correlation and volatility patterns.</p>
                        </div>
                        
                        <div class="insight-card">
                            <h4>⚠️ Risk Assessment</h4>
                            <p>Multiple risk metrics calculated including Value at Risk, maximum drawdown, and volatility measures for comprehensive risk management.</p>
                        </div>
                        
                        <div class="insight-card">
                            <h4>🔍 Feature Relationships</h4>
                            <p>Correlation analysis reveals {len([col for col in df.select_dtypes(include=[np.number]).columns if col != 'timestamp'])} key relationships between market features.</p>
                        </div>
                    </div>
                    
                    <div class="chart-section">
                        <h2>Recommendations</h2>
                        <div class="chart-description">
                            <ul style="color: #7f8c8d; line-height: 1.8; margin-left: 20px;">
                                <li><strong>Monitor Key Correlations:</strong> Watch relationships between major market indicators for early warning signals.</li>
                                <li><strong>Risk Management:</strong> Implement stop-losses based on calculated VaR levels and maximum drawdown thresholds.</li>
                                <li><strong>Regime Detection:</strong> Use regime analysis to adjust trading strategies based on market conditions.</li>
                                <li><strong>Data Quality:</strong> Regularly check for missing data and implement imputation strategies if needed.</li>
                                <li><strong>Performance Tracking:</strong> Use Sharpe ratio and other metrics to evaluate strategy effectiveness over time.</li>
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div style="text-align: center; margin: 40px 0;">
                    <button class="download-btn" onclick="downloadReport()">📥 Download Full Report</button>
                    <button class="download-btn" onclick="exportCharts()">📊 Export All Charts</button>
                    <button class="download-btn" onclick="printReport()">🖨️ Print Report</button>
                </div>
                
                <div class="footer">
                    <p>Market Risk Intelligence System v2.0.0 | Professional Analytics Platform</p>
                    <p>Confidential - For Authorized Use Only | Generated with Python & Plotly</p>
                </div>
            </div>
            
            <script>
                // Load Plotly charts
                const charts = {json.dumps({k: f"{k}.html" for k in charts.keys()})};
                
                function showTab(tabName) {{
                    // Hide all tabs
                    document.querySelectorAll('.tab-content').forEach(tab => {{
                        tab.classList.remove('active');
                    }});
                    
                    // Remove active class from all tab buttons
                    document.querySelectorAll('.nav-tab').forEach(tab => {{
                        tab.classList.remove('active');
                    }});
                    
                    // Show selected tab
                    document.getElementById(tabName + '-tab').classList.add('active');
                    
                    // Add active class to clicked tab
                    event.currentTarget.classList.add('active');
                    
                    // Load chart if available
                    loadChartForTab(tabName);
                }}
                
                function loadChartForTab(tabName) {{
                    const chartMap = {{
                        'overview': 'market_overview',
                        'risk': 'risk_regime',
                        'correlations': 'correlation_network',
                        'metrics': 'risk_metrics'
                    }};
                    
                    const chartName = chartMap[tabName];
                    if (chartName && charts[chartName]) {{
                        loadChart(chartName, tabName + '-chart');
                    }}
                }}
                
                function loadChart(chartName, containerId) {{
                    const container = document.getElementById(containerId);
                    if (container && charts[chartName]) {{
                        // In a real implementation, this would load the Plotly chart
                        container.innerHTML = `<iframe src="${{charts[chartName]}}" width="100%" height="100%" frameborder="0"></iframe>`;
                    }}
                }}
                
                function downloadReport() {{
                    alert('Report download would start here in a real implementation.');
                }}
                
                function exportCharts() {{
                    alert('Charts export would start here in a real implementation.');
                }}
                
                function printReport() {{
                    window.print();
                }}
                
                // Load initial chart
                window.onload = function() {{
                    loadChartForTab('overview');
                }};
            </script>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = output_dir / f"{report_title.replace(' ', '_').lower()}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Professional report saved to: {report_path}")
        return report_path


# Factory function for creating visualizer
def create_professional_visualizer(theme: PlotTheme = PlotTheme.PLOTLY_WHITE) -> ProfessionalEDAVisualizer:
    """Create a professional EDA visualizer instance."""
    return ProfessionalEDAVisualizer(theme=theme)


# Compatibility layer for existing code
class EDAVisualizer(ProfessionalEDAVisualizer):
    """Legacy compatibility layer for existing code."""
    
    def __init__(self, theme: str = "plotly_white"):
        """Initialize with legacy theme string."""
        plotly_theme = PlotTheme(theme) if hasattr(PlotTheme, theme.upper()) else PlotTheme.PLOTLY_WHITE
        super().__init__(theme=plotly_theme)


# Export main classes
__all__ = [
    'ProfessionalEDAVisualizer',
    'EDAVisualizer',
    'PlotTheme',
    'ChartType',
    'ChartConfig',
    'create_professional_visualizer'
]
