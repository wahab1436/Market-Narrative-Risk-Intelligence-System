"""
SHAP analysis for model explainability.
"""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader


class SHAPAnalyzer:
    """
    SHAP analysis for model interpretability.
    """
    
    def __init__(self):
        """Initialize SHAP analyzer."""
        self.config = config_loader.get_config("config")
        model_logger.info("SHAPAnalyzer initialized")
    
    def analyze_linear_model(self, model, X: pd.DataFrame, model_name: str = "linear") -> Dict:
        """
        Perform SHAP analysis for linear models.
        
        Args:
            model: Trained linear model
            X: Feature DataFrame
            model_name: Name of the model
        
        Returns:
            Dictionary with SHAP results
        """
        model_logger.info(f"Performing SHAP analysis for {model_name}")
        
        # Create explainer
        explainer = shap.Explainer(model, X)
        
        # Calculate SHAP values
        shap_values = explainer(X)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        summary_plot_path = Path(f"logs/shap_summary_{model_name}.png")
        plt.tight_layout()
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        bar_plot_path = Path(f"logs/shap_bar_{model_name}.png")
        plt.tight_layout()
        plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate feature importance
        shap_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        results = {
            'shap_values': shap_values,
            'feature_importance': shap_importance,
            'summary_plot_path': summary_plot_path,
            'bar_plot_path': bar_plot_path
        }
        
        return results
    
    def analyze_neural_network(self, model, X: np.ndarray, feature_names: List[str],
                             model_name: str = "neural_network") -> Dict:
        """
        Perform SHAP analysis for neural networks.
        
        Args:
            model: Trained neural network model
            X: Feature array
            feature_names: List of feature names
            model_name: Name of the model
        
        Returns:
            Dictionary with SHAP results
        """
        model_logger.info(f"Performing SHAP analysis for {model_name}")
        
        # Create explainer
        explainer = shap.KernelExplainer(
            lambda x: model.predict(x, verbose=0).flatten(),
            shap.kmeans(X, 10)
        )
        
        # Calculate SHAP values for a sample
        shap_values = explainer.shap_values(X[:100], nsamples=100)
        
        # Create feature DataFrame for plotting
        X_df = pd.DataFrame(X[:100], columns=feature_names)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_df, show=False)
        summary_plot_path = Path(f"logs/shap_summary_{model_name}.png")
        plt.tight_layout()
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate feature importance
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        results = {
            'shap_values': shap_values,
            'feature_importance': shap_importance,
            'summary_plot_path': summary_plot_path
        }
        
        return results
    
    def analyze_xgboost(self, model, X: pd.DataFrame, model_name: str = "xgboost") -> Dict:
        """
        Perform SHAP analysis for XGBoost models.
        
        Args:
            model: Trained XGBoost model
            X: Feature DataFrame
            model_name: Name of the model
        
        Returns:
            Dictionary with SHAP results
        """
        model_logger.info(f"Performing SHAP analysis for {model_name}")
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        summary_plot_path = Path(f"logs/shap_summary_{model_name}.png")
        plt.tight_layout()
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Dependence plots for top features
        top_features = X.columns[np.argsort(np.abs(shap_values).mean(axis=0))[-3:]]
        dependence_plots = {}
        
        for feature in top_features:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                feature,
                shap_values,
                X,
                show=False
            )
            plot_path = Path(f"logs/shap_dependence_{model_name}_{feature}.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            dependence_plots[feature] = plot_path
        
        # Calculate feature importance
        shap_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        results = {
            'shap_values': shap_values,
            'feature_importance': shap_importance,
            'summary_plot_path': summary_plot_path,
            'dependence_plots': dependence_plots
        }
        
        return results
    
    def generate_report(self, shap_results: Dict, output_path: Path) -> str:
        """
        Generate HTML report with SHAP visualizations.
        
        Args:
            shap_results: Dictionary with SHAP results
            output_path: Path to save HTML report
        
        Returns:
            HTML report content
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SHAP Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }
                h2 { color: #555; margin-top: 30px; }
                .plot-container { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }
                .feature-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .feature-table th, .feature-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                .feature-table th { background-color: #f2f2f2; }
                .plot-img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>SHAP Analysis Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <h2>Top Features by SHAP Importance</h2>
            {feature_table}
            
            <h2>SHAP Summary Plot</h2>
            <div class="plot-container">
                <img src="{summary_plot}" alt="SHAP Summary Plot" class="plot-img">
            </div>
            
            {dependence_plots}
        </body>
        </html>
        """
        
        # Prepare feature table
        if 'feature_importance' in shap_results:
            feature_df = shap_results['feature_importance'].head(10)
            feature_table = """
            <table class="feature-table">
                <tr>
                    <th>Feature</th>
                    <th>Mean |SHAP|</th>
                    <th>Importance (%)</th>
                </tr>
            """
            
            total_importance = feature_df['mean_abs_shap'].sum()
            for _, row in feature_df.iterrows():
                importance_pct = (row['mean_abs_shap'] / total_importance * 100) if total_importance > 0 else 0
                feature_table += f"""
                <tr>
                    <td>{row['feature']}</td>
                    <td>{row['mean_abs_shap']:.4f}</td>
                    <td>{importance_pct:.1f}%</td>
                </tr>
                """
            
            feature_table += "</table>"
        else:
            feature_table = "<p>No feature importance data available.</p>"
        
        # Prepare dependence plots section
        dependence_html = ""
        if 'dependence_plots' in shap_results:
            dependence_html = "<h2>SHAP Dependence Plots</h2>"
            for feature, plot_path in shap_results['dependence_plots'].items():
                dependence_html += f"""
                <div class="plot-container">
                    <h3>Feature: {feature}</h3>
                    <img src="{plot_path}" alt="Dependence Plot for {feature}" class="plot-img">
                </div>
                """
        
        # Fill template
        from datetime import datetime
        html_content = html_content.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            feature_table=feature_table,
            summary_plot=shap_results.get('summary_plot_path', ''),
            dependence_plots=dependence_html
        )
        
        # Save HTML report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        model_logger.info(f"SHAP report saved to {output_path}")
        return html_content
