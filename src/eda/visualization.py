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
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import model_logger


class EDAVisualizer:
    """
    Visualization tools for exploratory data analysis.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        """Initialize EDA visualizer."""
        self.theme = theme
        self.color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        model_logger.info("EDAVisualizer initialized")
    
    def set_theme(self):
        """Set plotting theme."""
        if self.theme == "seaborn":
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['font.size'] = 12
    
    def plot_feature_distributions(self, df: pd.DataFrame, 
                                 n_cols: int = 4,
                                 output_path: Optional[Path] = None) -> go.Figure:
        """
        Plot distributions of all numerical features.
        
        Args:
            df: Input DataFrame
            n_cols: Number of columns in subplot grid
            output_path: Path to save plot
        
        Returns:
            Plotly Figure object
        """
        # Select numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to top 20 features for readability
        if len(numeric_cols) > 20:
            # Select features with highest variance
            variances = df[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = variances.head(20).index.tolist()
        
        n_features = len(numeric_cols)
        n_rows = int(np.ceil(n_features / n_cols))
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols + 1
            col_num = idx % n_cols + 1
            
            # Get data
            data = df[col].dropna()
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=col,
                    nbinsx=min(50, len(data) // 10),
                    marker_color=self.color_palette[idx % len(self.color_palette)],
                    opacity=0.7,
                    histnorm='probability density'
                ),
                row=row, col=col_num
            )
            
            # Add statistics as annotation
            mean_val = data.mean()
            std_val = data.std()
            
            fig.add_annotation(
                xref=f"x{idx+1}",
                yref=f"y{idx+1}",
                x=0.98,
                y=0.98,
                xanchor="right",
                yanchor="top",
                text=f"μ={mean_val:.2f}<br>σ={std_val:.2f}",
                showarrow=False,
                font=dict(size=10),
                row=row, col=col_num
            )
        
        fig.update_layout(
            title_text="Feature Distributions",
            height=300 * n_rows,
            showlegend=False,
            template=self.theme
        )
        
        # Update axes
        for i in range(1, n_features + 1):
            fig.update_xaxes(title_text="Value", row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
            fig.update_yaxes(title_text="Density", row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
        
        if output_path:
            fig.write_image(output_path, width=1600, height=300*n_rows)
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                              method: str = 'pearson',
                              output_path: Optional[Path] = None) -> go.Figure:
        """
        Plot correlation matrix of numerical features.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            output_path: Path to save plot
        
        Returns:
            Plotly Figure object
        """
        # Select numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to top 30 features
        if len(numeric_cols) > 30:
            # Select features with highest variance
            variances = df[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = variances.head(30).index.tolist()
        
        if len(numeric_cols) < 2:
            model_logger.warning("Not enough numeric columns for correlation matrix")
            return None
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr(method=method)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hoverinfo='text',
            hovertemplate='Feature 1: %{y}<br>Feature 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        # Add annotations for high correlations
        high_corr_threshold = 0.7
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.index)):
                corr_value = corr_matrix.iloc[i, j]
                if i != j and abs(corr_value) > high_corr_threshold:
                    fig.add_annotation(
                        x=corr_matrix.columns[j],
                        y=corr_matrix.index[i],
                        text=f"{corr_value:.2f}",
                        showarrow=False,
                        font=dict(
                            size=12,
                            color="white" if abs(corr_value) > 0.8 else "black"
                        )
                    )
        
        fig.update_layout(
            title_text=f"Feature Correlation Matrix ({method.capitalize()})",
            height=800,
            width=1000,
            xaxis_title="Features",
            yaxis_title="Features",
            xaxis_tickangle=-45,
            template=self.theme
        )
        
        if output_path:
            fig.write_image(output_path, width=1200, height=800)
        
        return fig
    
    def plot_time_series_decomposition(self, df: pd.DataFrame, 
                                     target_col: str = 'weighted_stress_score',
                                     date_col: str = 'timestamp',
                                     period: int = 7,
                                     output_path: Optional[Path] = None) -> go.Figure:
        """
        Decompose time series into trend, seasonality, and residuals.
        
        Args:
            df: Input DataFrame
            target_col: Target column to decompose
            date_col: Date column name
            period: Seasonality period
            output_path: Path to save plot
        
        Returns:
            Plotly Figure object
        """
        if date_col not in df.columns or target_col not in df.columns:
            model_logger.warning(f"Required columns not found: {date_col} or {target_col}")
            return None
        
        # Ensure sorted by date
        df_sorted = df.sort_values(date_col).copy()
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
        df_sorted.set_index(date_col, inplace=True)
        
        # Resample to daily frequency
        daily_series = df_sorted[target_col].resample('D').mean()
        
        if len(daily_series) < period * 2:
            model_logger.warning(f"Insufficient data for decomposition. Need at least {period*2} days, got {len(daily_series)}")
            return None
        
        # Fill missing values for decomposition
        daily_series_filled = daily_series.interpolate(method='linear').ffill().bfill()
        
        # Simple decomposition
        # Trend: centered moving average
        trend = daily_series_filled.rolling(window=period, center=True).mean()
        
        # Detrend
        detrended = daily_series_filled - trend
        
        # Simple seasonality extraction
        seasonal = pd.Series(index=daily_series_filled.index, dtype=float)
        residuals = pd.Series(index=daily_series_filled.index, dtype=float)
        
        # For each position in season
        for i in range(period):
            mask = (daily_series_filled.index.dayofweek if period == 7 else 
                   daily_series_filled.index.day % period == i)
            seasonal[mask] = detrended[mask].mean()
        
        # Calculate residuals
        residuals = detrended - seasonal
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                f'Original: {target_col}',
                f'Trend ({period}-day moving average)',
                f'Seasonality (period={period})',
                'Residuals'
            ],
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Original
        fig.add_trace(
            go.Scatter(
                x=daily_series_filled.index,
                y=daily_series_filled.values,
                mode='lines',
                name='Original',
                line=dict(color=self.color_palette[0], width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.1)'
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
                line=dict(color=self.color_palette[1], width=3)
            ),
            row=2, col=1
        )
        
        # Seasonality
        fig.add_trace(
            go.Scatter(
                x=seasonal.index,
                y=seasonal.values,
                mode='lines',
                name='Seasonality',
                line=dict(color=self.color_palette[2], width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.1)'
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
                line=dict(color=self.color_palette[3], width=1),
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.1)'
            ),
            row=4, col=1
        )
        
        # Add zero line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
        
        # Calculate statistics
        stats_text = f"""
        <b>Statistics:</b><br>
        Original Mean: {daily_series_filled.mean():.3f}<br>
        Original Std: {daily_series_filled.std():.3f}<br>
        Trend Explained: {(trend.var() / daily_series_filled.var()):.1%}<br>
        Seasonality Explained: {(seasonal.var() / daily_series_filled.var()):.1%}<br>
        Residual Std: {residuals.std():.3f}
        """
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        fig.update_layout(
            height=1000,
            title_text=f"Time Series Decomposition: {target_col}",
            showlegend=True,
            hovermode='x unified',
            template=self.theme
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonality", row=3, col=1)
        fig.update_yaxes(title_text="Residuals", row=4, col=1)
        
        if output_path:
            fig.write_image(output_path, width=1400, height=1000)
        
        return fig
    
    def plot_feature_importance_comparison(self, feature_importance_dict: Dict[str, pd.DataFrame],
                                         top_n: int = 10,
                                         output_path: Optional[Path] = None) -> go.Figure:
        """
        Compare feature importance across different models.
        
        Args:
            feature_importance_dict: Dictionary of {model_name: feature_importance_df}
            top_n: Number of top features to show per model
            output_path: Path to save plot
        
        Returns:
            Plotly Figure object
        """
        if not feature_importance_dict:
            model_logger.warning("No feature importance data provided")
            return None
        
        # Prepare data for plotting
        plot_data = []
        
        for model_name, df in feature_importance_dict.items():
            if df.empty:
                continue
            
            # Determine importance column name
            importance_col = None
            for col in ['importance', 'coefficient', 'mean_abs_shap']:
                if col in df.columns:
                    importance_col = col
                    break
            
            if importance_col is None:
                continue
            
            # Get top N features
            top_df = df.head(top_n).copy()
            
            for _, row in top_df.iterrows():
                plot_data.append({
                    'model': model_name.replace('_', ' ').title(),
                    'feature': row['feature'],
                    'importance': abs(row[importance_col]),
                    'sign': np.sign(row[importance_col]) if importance_col == 'coefficient' else 1
                })
        
        if not plot_data:
            return None
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar chart
        fig = px.bar(
            plot_df,
            x='feature',
            y='importance',
            color='model',
            barmode='group',
            title=f'Top {top_n} Feature Importance Comparison Across Models',
            color_discrete_sequence=self.color_palette,
            hover_data=['sign']
        )
        
        # Add sign indicator for coefficients
        if 'sign' in plot_df.columns and plot_df['sign'].nunique() > 1:
            # Update hover template to show sign
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Model: %{customdata[0]}<br>Importance: %{y:.4f}<br>Sign: %{customdata[1]:+d}<extra></extra>'
            )
        
        fig.update_layout(
            height=600,
            xaxis_title="Feature",
            yaxis_title="Importance (absolute value)",
            xaxis_tickangle=-45,
            template=self.theme,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        if output_path:
            fig.write_image(output_path, width=1400, height=600)
        
        return fig
    
    def plot_stress_score_evolution(self, df: pd.DataFrame,
                                  date_col: str = 'timestamp',
                                  stress_col: str = 'weighted_stress_score',
                                  window: int = 7,
                                  output_path: Optional[Path] = None) -> go.Figure:
        """
        Plot stress score evolution over time with moving statistics.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            stress_col: Stress score column name
            window: Moving average window
            output_path: Path to save plot
        
        Returns:
            Plotly Figure object
        """
        if date_col not in df.columns or stress_col not in df.columns:
            model_logger.warning(f"Required columns not found: {date_col} or {stress_col}")
            return None
        
        # Prepare data
        plot_df = df.sort_values(date_col).copy()
        plot_df[date_col] = pd.to_datetime(plot_df[date_col])
        
        # Calculate moving statistics
        plot_df['moving_avg'] = plot_df[stress_col].rolling(window=window, center=True).mean()
        plot_df['moving_std'] = plot_df[stress_col].rolling(window=window, center=True).std()
        plot_df['upper_band'] = plot_df['moving_avg'] + plot_df['moving_std']
        plot_df['lower_band'] = plot_df['moving_avg'] - plot_df['moving_std']
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=['Stress Score Evolution', 'Daily Change'],
            shared_xaxes=True
        )
        
        # Main stress score plot
        fig.add_trace(
            go.Scatter(
                x=plot_df[date_col],
                y=plot_df[stress_col],
                mode='lines+markers',
                name='Stress Score',
                line=dict(color=self.color_palette[0], width=1),
                marker=dict(size=4),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Moving average
        fig.add_trace(
            go.Scatter(
                x=plot_df[date_col],
                y=plot_df['moving_avg'],
                mode='lines',
                name=f'{window}-day Moving Average',
                line=dict(color=self.color_palette[1], width=3)
            ),
            row=1, col=1
        )
        
        # Confidence band
        fig.add_trace(
            go.Scatter(
                x=plot_df[date_col].tolist() + plot_df[date_col].tolist()[::-1],
                y=plot_df['upper_band'].tolist() + plot_df['lower_band'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255, 127, 14, 0)'),
                name='±1 Std Dev',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Daily change plot
        plot_df['daily_change'] = plot_df[stress_col].diff()
        
        # Color bars based on direction
        colors = ['red' if x < 0 else 'green' for x in plot_df['daily_change']]
        
        fig.add_trace(
            go.Bar(
                x=plot_df[date_col],
                y=plot_df['daily_change'],
                name='Daily Change',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Calculate statistics
        total_days = len(plot_df)
        high_stress_days = (plot_df[stress_col] > plot_df[stress_col].quantile(0.75)).sum()
        stress_increase_days = (plot_df['daily_change'] > 0).sum()
        
        stats_text = f"""
        <b>Statistics:</b><br>
        Total Days: {total_days}<br>
        High Stress Days: {high_stress_days} ({high_stress_days/total_days:.1%})<br>
        Stress Increase Days: {stress_increase_days} ({stress_increase_days/total_days:.1%})<br>
        Current Stress: {plot_df[stress_col].iloc[-1]:.3f}
        """
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        fig.update_layout(
            height=800,
            title_text=f"Stress Score Evolution Analysis",
            showlegend=True,
            hovermode='x unified',
            template=self.theme
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Stress Score", row=1, col=1)
        fig.update_yaxes(title_text="Daily Change", row=2, col=1)
        
        if output_path:
            fig.write_image(output_path, width=1400, height=800)
        
        return fig
    
    def plot_model_performance_comparison(self, performance_dict: Dict[str, Dict],
                                        output_path: Optional[Path] = None) -> go.Figure:
        """
        Compare model performance across different metrics.
        
        Args:
            performance_dict: Dictionary of {model_name: {metric: value}}
            output_path: Path to save plot
        
        Returns:
            Plotly Figure object
        """
        if not performance_dict:
            model_logger.warning("No performance data provided")
            return None
        
        # Extract metrics
        metrics = set()
        for model_perf in performance_dict.values():
            metrics.update(model_perf.keys())
        
        # Convert to DataFrame
        plot_data = []
        for model_name, model_perf in performance_dict.items():
            for metric, value in model_perf.items():
                if isinstance(value, (int, float)):
                    plot_data.append({
                        'model': model_name.replace('_', ' ').title(),
                        'metric': metric.upper(),
                        'value': value
                    })
        
        if not plot_data:
            return None
        
        plot_df = pd.DataFrame(plot_data)
        
        # Pivot for heatmap
        pivot_df = plot_df.pivot(index='model', columns='metric', values='value')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis',
            text=pivot_df.round(3).values,
            texttemplate='%{text}',
            textfont={"size": 11},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            height=400,
            width=800,
            xaxis_title="Metric",
            yaxis_title="Model",
            template=self.theme
        )
        
        if output_path:
            fig.write_image(output_path, width=1000, height=500)
        
        return fig
    
    def create_eda_report(self, df: pd.DataFrame, output_dir: Path, 
                         report_name: str = "eda_report") -> Path:
        """
        Create comprehensive EDA report with all visualizations.
        
        Args:
            df: Input DataFrame
            output_dir: Directory to save report
            report_name: Name of the report
        
        Returns:
            Path to the generated report
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_logger.info(f"Creating EDA report in {output_dir}")
        
        # Generate all plots
        plots = {}
        
        try:
            # 1. Feature distributions
            plots['distributions'] = self.plot_feature_distributions(
                df, output_path=output_dir / 'feature_distributions.png'
            )
        except Exception as e:
            model_logger.error(f"Failed to create distributions plot: {e}")
        
        try:
            # 2. Correlation matrix
            plots['correlation'] = self.plot_correlation_matrix(
                df, output_path=output_dir / 'correlation_matrix.png'
            )
        except Exception as e:
            model_logger.error(f"Failed to create correlation matrix: {e}")
        
        try:
            # 3. Time series decomposition (if timestamp exists)
            if 'timestamp' in df.columns and 'weighted_stress_score' in df.columns:
                plots['decomposition'] = self.plot_time_series_decomposition(
                    df, output_path=output_dir / 'time_series_decomposition.png'
                )
        except Exception as e:
            model_logger.error(f"Failed to create time series decomposition: {e}")
        
        try:
            # 4. Stress score evolution
            if 'timestamp' in df.columns and 'weighted_stress_score' in df.columns:
                plots['stress_evolution'] = self.plot_stress_score_evolution(
                    df, output_path=output_dir / 'stress_score_evolution.png'
                )
        except Exception as e:
            model_logger.error(f"Failed to create stress evolution plot: {e}")
        
        # Create HTML report
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Exploratory Data Analysis Report - Market Narrative Risk Intelligence</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                .container { 
                    max-width: 1400px; 
                    margin: 0 auto; 
                    background: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                }
                .header { 
                    text-align: center; 
                    margin-bottom: 40px; 
                    padding-bottom: 20px; 
                    border-bottom: 3px solid #667eea;
                }
                h1 { 
                    color: #333; 
                    margin: 0; 
                    font-size: 2.5em;
                }
                .subtitle { 
                    color: #666; 
                    font-size: 1.1em; 
                    margin-top: 10px;
                }
                h2 { 
                    color: #444; 
                    margin-top: 40px; 
                    padding-bottom: 10px; 
                    border-bottom: 2px solid #e0e0e0;
                    font-size: 1.8em;
                }
                .plot-container { 
                    margin: 25px 0; 
                    padding: 25px; 
                    background: #f8f9fa; 
                    border-radius: 8px; 
                    border: 1px solid #e0e0e0;
                    transition: transform 0.2s;
                }
                .plot-container:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                }
                .plot-title { 
                    color: #555; 
                    margin-bottom: 15px; 
                    font-size: 1.3em;
                    display: flex;
                    align-items: center;
                }
                .plot-title::before {
                    content: "📊";
                    margin-right: 10px;
                }
                .plot-img { 
                    max-width: 100%; 
                    height: auto; 
                    border-radius: 5px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                }
                .stats-section { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                    margin: 30px 0;
                }
                .stats-card { 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 8px; 
                    border-left: 4px solid #667eea;
                }
                .stats-card h3 { 
                    margin-top: 0; 
                    color: #444;
                }
                table { 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0;
                }
                th, td { 
                    padding: 12px 15px; 
                    text-align: left; 
                    border-bottom: 1px solid #ddd;
                }
                th { 
                    background-color: #667eea; 
                    color: white; 
                    font-weight: 600;
                }
                tr:hover { 
                    background-color: #f5f5f5;
                }
                .info-box { 
                    background: #e3f2fd; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 20px 0;
                    border-left: 4px solid #2196f3;
                }
                .footer { 
                    text-align: center; 
                    margin-top: 50px; 
                    color: #666; 
                    font-size: 0.9em;
                    padding-top: 20px;
                    border-top: 1px solid #e0e0e0;
                }
                .badge {
                    display: inline-block;
                    padding: 3px 10px;
                    border-radius: 12px;
                    font-size: 0.8em;
                    font-weight: 600;
                    margin-left: 10px;
                }
                .badge-success { background: #4caf50; color: white; }
                .badge-warning { background: #ff9800; color: white; }
                .badge-danger { background: #f44336; color: white; }
                .nav-tabs {
                    display: flex;
                    border-bottom: 2px solid #e0e0e0;
                    margin-bottom: 20px;
                }
                .nav-tab {
                    padding: 10px 20px;
                    cursor: pointer;
                    border: none;
                    background: none;
                    font-size: 1em;
                    color: #666;
                    border-bottom: 3px solid transparent;
                    transition: all 0.3s;
                }
                .nav-tab.active {
                    color: #667eea;
                    border-bottom: 3px solid #667eea;
                    font-weight: 600;
                }
                .tab-content {
                    display: none;
                }
                .tab-content.active {
                    display: block;
                }
                @media (max-width: 768px) {
                    .container { padding: 15px; }
                    .stats-section { grid-template-columns: 1fr; }
                    h1 { font-size: 2em; }
                }
            </style>
            <script>
                function showTab(tabName) {
                    // Hide all tab contents
                    document.querySelectorAll('.tab-content').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    
                    // Remove active class from all tabs
                    document.querySelectorAll('.nav-tab').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    
                    // Show selected tab content
                    document.getElementById(tabName + '-tab').classList.add('active');
                    
                    // Add active class to clicked tab
                    event.currentTarget.classList.add('active');
                }
                
                function downloadPlot(plotName) {
                    alert('Downloading ' + plotName + '...');
                    // In a real implementation, this would trigger file download
                }
            </script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Exploratory Data Analysis Report</h1>
                    <div class="subtitle">
                        Market Narrative Risk Intelligence System
                    </div>
                    <div class="subtitle">
                        Generated: {timestamp}
                    </div>
                </div>
                
                <div class="info-box">
                    <strong>📈 Report Summary:</strong> This report provides comprehensive analysis of the market narrative data, 
                    including feature distributions, correlations, time series patterns, and stress score evolution.
                </div>
                
                <div class="stats-section">
                    <div class="stats-card">
                        <h3>Dataset Overview</h3>
                        <p><strong>Shape:</strong> {shape}</p>
                        <p><strong>Time Range:</strong> {time_range}</p>
                        <p><strong>Total Features:</strong> {n_features}</p>
                        <p><strong>Numeric Features:</strong> {n_numeric}</p>
                        <p><strong>Categorical Features:</strong> {n_categorical}</p>
                    </div>
                    
                    <div class="stats-card">
                        <h3>Data Quality</h3>
                        <p><strong>Missing Values:</strong> {missing_pct}%</p>
                        <p><strong>Duplicate Rows:</strong> {duplicates}</p>
                        <p><strong>Data Types:</strong> {data_types}</p>
                        <p><strong>Memory Usage:</strong> {memory_usage}</p>
                    </div>
                    
                    <div class="stats-card">
                        <h3>Stress Score Statistics</h3>
                        {stress_stats}
                    </div>
                </div>
                
                <div class="nav-tabs">
                    <button class="nav-tab active" onclick="showTab('visualizations')">Visualizations</button>
                    <button class="nav-tab" onclick="showTab('statistics')">Statistics</button>
                    <button class="nav-tab" onclick="showTab('correlations')">Correlations</button>
                    <button class="nav-tab" onclick="showTab('timeseries')">Time Series</button>
                </div>
                
                <div id="visualizations-tab" class="tab-content active">
                    {visualizations}
                </div>
                
                <div id="statistics-tab" class="tab-content">
                    <h2>Statistical Summary</h2>
                    {summary_table}
                    
                    <h2>Missing Values Analysis</h2>
                    {missing_table}
                </div>
                
                <div id="correlations-tab" class="tab-content">
                    <h2>Feature Correlations</h2>
                    <p>Analysis of relationships between different features in the dataset.</p>
                    {correlation_insights}
                </div>
                
                <div id="timeseries-tab" class="tab-content">
                    <h2>Time Series Analysis</h2>
                    <p>Analysis of temporal patterns and trends in the data.</p>
                    {timeseries_insights}
                </div>
                
                <div class="info-box">
                    <strong>💡 Insights & Recommendations:</strong>
                    <ul>
                        <li>Monitor features with high correlation to stress scores for early warning signals</li>
                        <li>Check for data quality issues in features with high missing values</li>
                        <li>Consider seasonality patterns when interpreting stress score trends</li>
                        <li>Validate model assumptions based on feature distributions</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Market Narrative Risk Intelligence System v1.0.0</p>
                    <p>Generated with Python, Plotly, and ❤️</p>
                    <p>Confidential - For Internal Use Only</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Prepare statistics
        shape = f"{df.shape[0]:,} rows × {df.shape[1]:,} columns"
        
        # Time range
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
        else:
            time_range = "Not available"
        
        # Feature counts
        n_numeric = len(df.select_dtypes(include=[np.number]).columns)
        n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        n_features = n_numeric + n_categorical
        
        # Missing values
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100).round(2)
        duplicates = df.duplicated().sum()
        
        # Data types
        data_types = df.dtypes.value_counts().to_dict()
        data_types_str = ', '.join([f"{k}: {v}" for k, v in data_types.items()])
        
        # Memory usage
        memory_usage = f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        
        # Stress score statistics
        stress_stats = ""
        if 'weighted_stress_score' in df.columns:
            stress_series = df['weighted_stress_score'].dropna()
            if len(stress_series) > 0:
                stress_stats = f"""
                <p><strong>Mean:</strong> {stress_series.mean():.3f}</p>
                <p><strong>Std Dev:</strong> {stress_series.std():.3f}</p>
                <p><strong>Min:</strong> {stress_series.min():.3f}</p>
                <p><strong>Max:</strong> {stress_series.max():.3f}</p>
                <p><strong>High Stress Days:</strong> {(stress_series > stress_series.quantile(0.75)).sum()}</p>
                """
        
        # Summary table
        summary_df = df.describe().round(3)
        summary_table = summary_df.to_html(classes='data-table', border=0)
        
        # Missing values table
        missing_df = df.isnull().sum().to_frame('missing_count')
        missing_df['missing_pct'] = (missing_df['missing_count'] / len(df) * 100).round(2)
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)
        missing_table = missing_df.head(20).to_html(classes='data-table', border=0)
        
        # Prepare visualizations HTML
        viz_html = ""
        
        plot_descriptions = {
            'distributions': 'Distribution of numerical features showing data spread and central tendencies.',
            'correlation': 'Correlation matrix highlighting relationships between different features.',
            'decomposition': 'Time series decomposition showing trend, seasonality, and residual components.',
            'stress_evolution': 'Evolution of stress scores over time with moving averages and volatility bands.'
        }
        
        for plot_name, fig in plots.items():
            if fig is not None:
                plot_path = output_dir / f'{plot_name}.png'
                description = plot_descriptions.get(plot_name, '')
                
                viz_html += f"""
                <div class="plot-container">
                    <div class="plot-title">
                        {plot_name.replace('_', ' ').title()}
                        <button onclick="downloadPlot('{plot_name}')" style="margin-left: auto; background: #667eea; color: white; border: none; padding: 5px 15px; border-radius: 4px; cursor: pointer; font-size: 0.9em;">
                            Download
                        </button>
                    </div>
                    <p>{description}</p>
                    <img src="{plot_path.name}" alt="{plot_name}" class="plot-img">
                </div>
                """
        
        # Correlation insights
        correlation_insights = ""
        if 'correlation' in plots and plots['correlation'] is not None:
            # Calculate top correlations
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)
                
                # Find top correlations
                top_corrs = []
                for col in corr_matrix.columns:
                    if col in corr_matrix.index:
                        max_corr = corr_matrix[col].max()
                        if max_corr > 0.5:
                            max_col = corr_matrix[col].idxmax()
                            top_corrs.append((col, max_col, max_corr))
                
                if top_corrs:
                    correlation_insights = "<h3>Top Feature Correlations (>0.5)</h3><ul>"
                    for col1, col2, corr in sorted(top_corrs, key=lambda x: x[2], reverse=True)[:10]:
                        correlation_insights += f"<li><strong>{col1}</strong> ↔ <strong>{col2}</strong>: {corr:.3f}</li>"
                    correlation_insights += "</ul>"
        
        # Time series insights
        timeseries_insights = ""
        if 'timestamp' in df.columns:
            timeseries_insights = """
            <div class="info-box">
                <strong>Time Series Analysis:</strong>
                <ul>
                    <li>Check for stationarity in the time series</li>
                    <li>Look for seasonality patterns (weekly, monthly)</li>
                    <li>Identify trend direction and strength</li>
                    <li>Detect structural breaks or regime changes</li>
                </ul>
            </div>
            """
        
        # Fill template
        from datetime import datetime
        html_content = html_content.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            shape=shape,
            time_range=time_range,
            n_features=n_features,
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            missing_pct=missing_pct,
            duplicates=duplicates,
            data_types=data_types_str,
            memory_usage=memory_usage,
            stress_stats=stress_stats,
            summary_table=summary_table,
            missing_table=missing_table,
            visualizations=viz_html,
            correlation_insights=correlation_insights,
            timeseries_insights=timeseries_insights
        )
        
        # Save HTML report
        report_path = output_dir / f'{report_name}.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Also save a JSON summary
        summary_data = {
            'generated_at': datetime.now().isoformat(),
            'dataset_shape': df.shape,
            'time_range': time_range,
            'feature_counts': {
                'total': n_features,
                'numeric': n_numeric,
                'categorical': n_categorical
            },
            'data_quality': {
                'missing_pct': float(missing_pct),
                'duplicates': int(duplicates),
                'memory_mb': float(memory_usage.split()[0])
            }
        }
        
        if 'weighted_stress_score' in df.columns:
            stress_series = df['weighted_stress_score'].dropna()
            summary_data['stress_score_stats'] = {
                'mean': float(stress_series.mean()),
                'std': float(stress_series.std()),
                'min': float(stress_series.min()),
                'max': float(stress_series.max()),
                'q25': float(stress_series.quantile(0.25)),
                'q75': float(stress_series.quantile(0.75))
            }
        
        import json
        with open(output_dir / 'eda_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        model_logger.info(f"EDA report saved to {report_path}")
        return report_path


# Factory function
def create_eda_visualizer(theme: str = "plotly_white") -> EDAVisualizer:
    """Create an EDA visualizer instance."""
    return EDAVisualizer(theme=theme)
