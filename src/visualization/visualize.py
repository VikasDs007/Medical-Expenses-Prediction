"""
Visualization functions for the medical insurance prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InsuranceVisualizer:
    """
    A comprehensive visualization class for insurance data analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_distribution(self, data: pd.DataFrame, column: str, 
                         title: Optional[str] = None, bins: int = 30) -> go.Figure:
        """
        Plot distribution of a numeric column.
        
        Args:
            data (pd.DataFrame): Dataset
            column (str): Column name to plot
            title (str, optional): Plot title
            bins (int): Number of bins for histogram
            
        Returns:
            go.Figure: Plotly figure
        """
        if title is None:
            title = f'Distribution of {column.title()}'
        
        fig = px.histogram(
            data, 
            x=column, 
            nbins=bins,
            title=title,
            marginal="box"
        )
        
        fig.update_layout(
            xaxis_title=column.title(),
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig
    
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                               title: str = "Correlation Matrix") -> go.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data (pd.DataFrame): Dataset
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            title=title,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            text_auto=True
        )
        
        fig.update_layout(
            width=800,
            height=600
        )
        
        return fig
    
    def plot_categorical_analysis(self, data: pd.DataFrame, cat_column: str, 
                                 target_column: str = 'charges') -> go.Figure:
        """
        Plot categorical variable analysis with target variable.
        
        Args:
            data (pd.DataFrame): Dataset
            cat_column (str): Categorical column name
            target_column (str): Target column name
            
        Returns:
            go.Figure: Plotly figure
        """
        fig = px.box(
            data, 
            x=cat_column, 
            y=target_column,
            title=f'{target_column.title()} by {cat_column.title()}',
            points="outliers"
        )
        
        fig.update_layout(
            xaxis_title=cat_column.title(),
            yaxis_title=target_column.title()
        )
        
        return fig
    
    def plot_scatter_analysis(self, data: pd.DataFrame, x_column: str, 
                             y_column: str, color_column: Optional[str] = None,
                             title: Optional[str] = None) -> go.Figure:
        """
        Plot scatter plot analysis.
        
        Args:
            data (pd.DataFrame): Dataset
            x_column (str): X-axis column
            y_column (str): Y-axis column
            color_column (str, optional): Column for color coding
            title (str, optional): Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        if title is None:
            title = f'{y_column.title()} vs {x_column.title()}'
        
        fig = px.scatter(
            data,
            x=x_column,
            y=y_column,
            color=color_column,
            title=title,
            trendline="ols"
        )
        
        fig.update_layout(
            xaxis_title=x_column.title(),
            yaxis_title=y_column.title()
        )
        
        return fig
    
    def plot_feature_importance(self, importance_data: pd.DataFrame, 
                               title: str = "Feature Importance") -> go.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_data (pd.DataFrame): Feature importance data
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        # Sort by importance
        importance_data = importance_data.sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_data,
            x='importance',
            y='feature',
            orientation='h',
            title=title
        )
        
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Features",
            height=max(400, len(importance_data) * 30)
        )
        
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]], 
                             metric: str = 'r2') -> go.Figure:
        """
        Plot model comparison.
        
        Args:
            results (Dict[str, Dict[str, float]]): Model results
            metric (str): Metric to compare
            
        Returns:
            go.Figure: Plotly figure
        """
        models = list(results.keys())
        values = [results[model][metric] for model in models]
        
        fig = px.bar(
            x=models,
            y=values,
            title=f'Model Comparison - {metric.upper()}',
            labels={'x': 'Models', 'y': metric.upper()}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45
        )
        
        return fig
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 title: str = "Predictions vs Actual") -> go.Figure:
        """
        Plot predictions vs actual values.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        fig = px.scatter(
            x=y_true,
            y=y_pred,
            title=title,
            labels={'x': 'Actual Values', 'y': 'Predicted Values'}
        )
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            )
        )
        
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = "Residual Plot") -> go.Figure:
        """
        Plot residuals.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        residuals = y_true - y_pred
        
        fig = px.scatter(
            x=y_pred,
            y=residuals,
            title=title,
            labels={'x': 'Predicted Values', 'y': 'Residuals'}
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        return fig
    
    def create_eda_dashboard(self, data: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Create a comprehensive EDA dashboard.
        
        Args:
            data (pd.DataFrame): Dataset
            
        Returns:
            Dict[str, go.Figure]: Dictionary of plots
        """
        plots = {}
        
        # Distribution of target variable
        plots['charges_distribution'] = self.plot_distribution(
            data, 'charges', 'Distribution of Insurance Charges'
        )
        
        # Correlation matrix
        plots['correlation_matrix'] = self.plot_correlation_matrix(data)
        
        # Categorical analyses
        categorical_columns = ['sex', 'smoker', 'region']
        for col in categorical_columns:
            if col in data.columns:
                plots[f'{col}_analysis'] = self.plot_categorical_analysis(data, col)
        
        # Scatter plots
        numeric_columns = ['age', 'bmi', 'children']
        for col in numeric_columns:
            if col in data.columns:
                plots[f'{col}_scatter'] = self.plot_scatter_analysis(
                    data, col, 'charges', 'smoker'
                )
        
        return plots
    
    def create_model_evaluation_dashboard(self, results: Dict[str, Dict[str, float]],
                                        y_true: np.ndarray, 
                                        predictions: Dict[str, np.ndarray]) -> Dict[str, go.Figure]:
        """
        Create model evaluation dashboard.
        
        Args:
            results (Dict[str, Dict[str, float]]): Model results
            y_true (np.ndarray): True values
            predictions (Dict[str, np.ndarray]): Model predictions
            
        Returns:
            Dict[str, go.Figure]: Dictionary of evaluation plots
        """
        plots = {}
        
        # Model comparison plots
        for metric in ['r2', 'mae', 'rmse']:
            plots[f'{metric}_comparison'] = self.plot_model_comparison(results, metric)
        
        # Prediction vs actual plots for each model
        for model_name, y_pred in predictions.items():
            plots[f'{model_name}_pred_vs_actual'] = self.plot_prediction_vs_actual(
                y_true, y_pred, f'{model_name}: Predictions vs Actual'
            )
            
            plots[f'{model_name}_residuals'] = self.plot_residuals(
                y_true, y_pred, f'{model_name}: Residual Plot'
            )
        
        return plots
    
    def save_plot(self, fig: go.Figure, filename: str, format: str = 'html') -> None:
        """
        Save a plot to file.
        
        Args:
            fig (go.Figure): Plotly figure
            filename (str): Output filename
            format (str): Output format ('html', 'png', 'pdf', 'svg')
        """
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'pdf':
            fig.write_image(filename)
        elif format == 'svg':
            fig.write_image(filename)
        else:
            raise ValueError("Format must be 'html', 'png', 'pdf', or 'svg'")

def create_summary_statistics_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a comprehensive summary statistics table.
    
    Args:
        data (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Summary statistics
    """
    # Numeric summary
    numeric_summary = data.describe()
    
    # Add additional statistics
    additional_stats = pd.DataFrame({
        col: {
            'missing': data[col].isnull().sum(),
            'missing_pct': (data[col].isnull().sum() / len(data)) * 100,
            'unique': data[col].nunique(),
            'skewness': data[col].skew() if data[col].dtype in ['int64', 'float64'] else np.nan,
            'kurtosis': data[col].kurtosis() if data[col].dtype in ['int64', 'float64'] else np.nan
        }
        for col in data.columns
    }).T
    
    # Combine summaries
    combined_summary = pd.concat([numeric_summary.T, additional_stats], axis=1)
    
    return combined_summary

def create_categorical_summary(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create summary for categorical variables.
    
    Args:
        data (pd.DataFrame): Dataset
        
    Returns:
        Dict[str, pd.DataFrame]: Categorical summaries
    """
    categorical_columns = data.select_dtypes(include=['object']).columns
    summaries = {}
    
    for col in categorical_columns:
        value_counts = data[col].value_counts()
        percentages = data[col].value_counts(normalize=True) * 100
        
        summary = pd.DataFrame({
            'Count': value_counts,
            'Percentage': percentages
        })
        
        summaries[col] = summary
    
    return summaries