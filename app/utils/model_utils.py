import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List, Tuple
import joblib
import os
import streamlit as st

class ModelTrainer:
    """
    A class to handle training and evaluation of multiple regression models.
    """
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        self.trained_models = {}
        self.results = {}
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all models and store them.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
            except Exception as e:
                st.warning(f"Failed to train {name}: {str(e)}")
        
        return self.trained_models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation results for each model
        """
        for name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                self.results[name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred
                }
            except Exception as e:
                st.warning(f"Failed to evaluate {name}: {str(e)}")
        
        return self.results
    
    def get_best_model(self, metric: str = 'r2') -> Tuple[str, Any]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric (str): Metric to use for comparison ('r2', 'mae', 'rmse')
            
        Returns:
            Tuple[str, Any]: (model_name, model_object)
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")
        
        if metric == 'r2':
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        elif metric in ['mae', 'rmse']:
            best_model_name = min(self.results.keys(), key=lambda x: self.results[x][metric])
        else:
            raise ValueError("Metric must be 'r2', 'mae', or 'rmse'")
        
        return best_model_name, self.trained_models[best_model_name]
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.trained_models[model_name], filepath)
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Any: Loaded model object
        """
        return joblib.load(filepath)

def calculate_prediction_intervals(model, X: pd.DataFrame, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals for a model (works best with ensemble methods).
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features for prediction
        confidence (float): Confidence level for intervals
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Lower and upper bounds of prediction intervals
    """
    if hasattr(model, 'estimators_'):
        # For ensemble methods like Random Forest
        predictions = np.array([estimator.predict(X) for estimator in model.estimators_])
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        z_score = 1.96  # For 95% confidence interval
        
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        return lower_bound, upper_bound
    else:
        # For other models, return simple prediction
        pred = model.predict(X)
        return pred, pred

def get_model_interpretation(model, feature_names: List[str]) -> Dict[str, Any]:
    """
    Get model interpretation information.
    
    Args:
        model: Trained model
        feature_names (List[str]): List of feature names
        
    Returns:
        Dict[str, Any]: Model interpretation data
    """
    interpretation = {}
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        interpretation['feature_importance'] = dict(zip(feature_names, importance))
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        interpretation['feature_importance'] = dict(zip(feature_names, importance))
    
    # Model parameters
    interpretation['model_params'] = model.get_params()
    
    return interpretation

@st.cache_data
def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison DataFrame of model results.
    
    Args:
        results (Dict[str, Dict[str, float]]): Model evaluation results
        
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'MAE': f"${metrics['mae']:,.0f}",
            'RMSE': f"${metrics['rmse']:,.0f}",
            'R² Score': f"{metrics['r2']:.3f}"
        })
    
    df = pd.DataFrame(comparison_data)
    return df.sort_values('R² Score', ascending=False)

def predict_with_model(model, preprocessor, input_data: Dict[str, Any]) -> float:
    """
    Make a prediction with a trained model and preprocessor.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        input_data (Dict[str, Any]): Input data for prediction
        
    Returns:
        float: Predicted value
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input
    X_processed = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(X_processed)[0]
    
    return prediction