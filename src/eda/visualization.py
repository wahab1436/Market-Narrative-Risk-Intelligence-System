"""
Exploratory Data Analysis visualization module.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Optional

from src.utils.logger import model_logger


class EDAVisualizer:
    """
    Visualization tools for exploratory data analysis.
    """
    
    def __init__(self):
        """Initialize EDA visualizer."""
        model_logger.info("EDAVisualizer initialized")
    
    def plot_feature_distributions(self, df: pd.DataFrame, output_path: Optional[Path] = None):
        """
        Plot distributions of all numerical features.
        
        Args:
            df: Input DataFrame
            output_path: Path to save plot
        """
        # Select numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to top 20 features for readability
        if len(numeric_cols) > 20:
            numeric_cols = numeric_cols[:20]
        
        n_cols = 4
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols
        )
        
        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols + 1
            col_num = idx % n_cols + 1
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    name=col,
                    nbinsx=50
                ),
                row=row, col=col_num
            )
        
        fig.update_layout(
            title_text="Feature Distributions",
            height=300 * n_rows,
            showlegend=False
        )
        
        if output_path:
            fig.write_image(output_path)
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, output_path: Optional[Path] = None):
        """
        Plot correlation matrix of numerical features.
        
        Args:
            df: Input DataFrame
            output_path: Path to save plot
        """
        # Select numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to top 30 features
        if len(numeric_cols) > 30:
            numeric_cols = numeric_cols[:30]
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title_text="Feature Correlation Matrix",
            height=600,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        if output_path:
            fig.write_image(output_path)
        
        return fig
    
    def plot_time_series_decomposition(self, df: pd.DataFrame, target_col: str, 
                                     date_col: str = 'timestamp', 
                                     output_path: Optional[Path] = None):
        """
        Decompose time series into trend, seasonality, and residuals.
        
        Args:
            df: Input DataFrame
            target_col: Target column to decompose
            date_col: Date column name
            output_path: Path to save plot
        """
        if date_col not in df.columns or target_col not in df.columns:
            return None
        
        # Ensure sorted by date
        df_sorted = df.sort_values(date_col).copy()
        df_sorted.set_index(date_col, inplace=True)
        
        # Resample to daily frequency
        daily_series = df_sorted[target_col].resample('D').mean().ffill()
        
        # Simple decomposition
        # Trend: 7-day rolling average
        trend = daily_series.rolling(window=7, center=True).mean()
        
        # Detrend
        detrended = daily_series - trend
        
        # Seasonality: Weekly pattern
        daily_series.index = pd.to_datetime(daily_series.index)
        detrended_df = pd.DataFrame({
            'value': detrended,
            'day_of_week': daily_series.index.dayofweek
        })
        
        seasonality = detrended_df.groupby('day_of_week')['value'].mean()
        
        # Residuals
        seasonal_component = pd.Series(
            [seasonality[dow] for dow in daily_series.index.dayofweek],
            index=daily_series.index
        )
        residuals = detrended - seasonal_component
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonality', 'Residuals'],
            vertical_spacing=0.1
        )
        
        # Original
        fig.add_trace(
            go.Scatter(
                x=daily_series.index,
                y=daily_series.values,
                mode='lines',
                name='Original'
            ),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(
                x=trend.index,
                y=trend.values,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Seasonality
        fig.add_trace(
            go.Scatter(
                x=seasonality.index,
                y=seasonality.values,
                mode='lines+markers',
                name='Seasonality',
                line=dict(color='green', width=2)
            ),
            row=3, col=1
        )
        
        # Residuals
        fig.add_trace(
            go.Scatter(
                x=residuals.index,
                y=residuals.values,
                mode='lines',
                name='Residuals',
                line=dict(color='orange', width=1)
            ),
            row=4, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text=f"Time Series Decomposition: {target_col}",
            showlegend=True
        )
        
        if output_path:
            fig.write_image(output_path)
        
        return fig
    
    def plot_feature_importance_comparison(self, feature_importance_dict: Dict[str, pd.DataFrame],
                                         output_path: Optional[Path] = None):
        """
        Compare feature importance across different models.
        
        Args:
            feature_importance_dict: Dictionary of {model_name: feature_importance_df}
            output_path: Path to save plot
        """
        # Combine top features from each model
        top_features = {}
        
        for model_name, df in feature_importance_dict.items():
            # Get top 10 features
            top_df = df.head(10).copy()
            top_df['model'] = model_name
            top_features[model_name] = top_df
        
        # Combine all
        combined_df = pd.concat(top_features.values(), ignore_index=True)
        
        # Create grouped bar chart
        fig = px.bar(
            combined_df,
            x='feature',
            y='importance' if 'importance' in combined_df.columns else 'coefficient',
            color='model',
            barmode='group',
            title='Feature Importance Comparison Across Models'
        )
        
        fig.update_layout(
            height=600,
            xaxis_title="Feature",
            yaxis_title="Importance",
            xaxis_tickangle=-45
        )
        
        if output_path:
            fig.write_image(output_path)
        
        return fig
    
    def create_eda_report(self, df: pd.DataFrame, output_dir: Path):
        """
        Create comprehensive EDA report with all visualizations.
        
        Args:
            df: Input DataFrame
            output_dir: Directory to save report
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        plots = {
            'distributions': self.plot_feature_distributions(
                df, output_dir / 'feature_distributions.png'
            ),
            'correlation': self.plot_correlation_matrix(
                df, output_dir / 'correlation_matrix.png'
            )
        }
        
        # Add time series decomposition if timestamp exists
        if 'timestamp' in df.columns and 'weighted_stress_score' in df.columns:
            plots['decomposition'] = self.plot_time_series_decomposition(
                df, 'weighted_stress_score', output_dir / 'time_series_decomposition.png'
            )
        
        # Create HTML report
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>EDA Report - Market Narrative Risk Intelligence</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }
                h2 { color: #555; margin-top: 30px; }
                .plot-container { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }
                .stats-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .stats-table th, .stats-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                .stats-table th { background-color: #f2f2f2; }
                .plot-img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Exploratory Data Analysis Report</h1>
            <p>Generated on: {timestamp}</p>
            <p>Dataset shape: {shape}</p>
            
            <h2>Dataset Summary</h2>
            {summary_table}
            
            <h2>Missing Values</h2>
            {missing_table}
            
            <h2>Visualizations</h2>
            {visualizations}
        </body>
        </html>
        """
        
        # Prepare statistics
        summary_stats = df.describe().round(3).to_html(classes='stats-table')
        missing_stats = df.isnull().sum().to_frame('missing_count').to_html(classes='stats-table')
        
        # Prepare visualizations HTML
        viz_html = ""
        for plot_name, fig in plots.items():
            plot_path = output_dir / f'{plot_name}.png'
            viz_html += f"""
            <div class="plot-container">
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="{plot_path.name}" alt="{plot_name}" class="plot-img">
            </div>
            """
        
        # Fill template
        from datetime import datetime
        html_content = html_content.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            shape=f"{df.shape[0]} rows × {df.shape[1]} columns",
            summary_table=summary_stats,
            missing_table=missing_stats,
            visualizations=viz_html
        )
        
        # Save HTML report
        report_path = output_dir / 'eda_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        model_logger.info(f"EDA report saved to {report_path}")
        return report_path
