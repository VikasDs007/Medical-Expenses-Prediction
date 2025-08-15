"""
Model training functions for the medical insurance prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
from typing import Dict, Any, Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A comprehensive model trainer for regression tasks.
    """
    
    def __init__(self):
        self.models = self._initialize_models()
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all models with default parameters."""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            ),
            'Support Vector Regression': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            )
        }
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, 
                          y_train: pd.Series) -> Any:
        """
        Train a single model.
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Any: Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        logger.info(f"Training {model_name}...")
        
        try:
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model
            logger.info(f"Successfully trained {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {str(e)}")
            raise
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        logger.info("Training all models...")
        
        for model_name in self.models.keys():
            try:
                self.train_single_model(model_name, X_train, y_train)
            except Exception as e:
                logger.warning(f"Skipping {model_name} due to error: {str(e)}")
        
        logger.info(f"Successfully trained {len(self.trained_models)} models")
        return self.trained_models
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred)
        }
        
        return metrics
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation results for all models
        """
        logger.info("Evaluating all models...")
        
        for model_name in self.trained_models.keys():
            try:
                metrics = self.evaluate_model(model_name, X_test, y_test)
                self.results[model_name] = metrics
                logger.info(f"{model_name} - R²: {metrics['r2']:.3f}, RMSE: ${metrics['rmse']:,.0f}")
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {str(e)}")
        
        return self.results
    
    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                           cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_name (str): Name of the model
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Cross-validation scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        cv_mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
        
        return {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std()
        }
    
    def hyperparameter_tuning(self, model_name: str, X_train: pd.DataFrame, 
                             y_train: pd.Series, param_grid: Dict[str, List], 
                             cv: int = 5) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name (str): Name of the model
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            param_grid (Dict[str, List]): Parameter grid for tuning
            cv (int): Number of cross-validation folds
            
        Returns:
            Any: Best model after tuning
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
        
        # Update the model with best parameters
        self.trained_models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def get_best_model(self, metric: str = 'r2') -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            Tuple[str, Any]: (model_name, model_object)
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")
        
        if metric == 'r2':
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        elif metric in ['mae', 'rmse', 'mse']:
            best_model_name = min(self.results.keys(), key=lambda x: self.results[x][metric])
        else:
            raise ValueError("Metric must be 'r2', 'mae', 'rmse', or 'mse'")
        
        self.best_model_name = best_model_name
        self.best_model = self.trained_models[best_model_name]
        
        return best_model_name, self.best_model
    
    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """
        Get feature importance from a trained model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            pd.DataFrame: Feature importance data
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            logger.warning(f"Model {model_name} doesn't have feature importance")
            return pd.DataFrame()
        
        # Assuming we have feature names stored somewhere
        # This would need to be passed or stored in the class
        feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        joblib.dump(self.trained_models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def save_all_models(self, directory: str) -> None:
        """
        Save all trained models to a directory.
        
        Args:
            directory (str): Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        for model_name in self.trained_models.keys():
            filename = f"{model_name.lower().replace(' ', '_')}.pkl"
            filepath = os.path.join(directory, filename)
            self.save_model(model_name, filepath)
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Any: Loaded model
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get a summary of all model results.
        
        Returns:
            pd.DataFrame: Summary of model performance
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, metrics in self.results.items():
            summary_data.append({
                'Model': model_name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2'],
                'MSE': metrics['mse']
            })
        
        df = pd.DataFrame(summary_data)
        return df.sort_values('R²', ascending=False)

def get_default_param_grids() -> Dict[str, Dict[str, List]]:
    """
    Get default parameter grids for hyperparameter tuning.
    
    Returns:
        Dict[str, Dict[str, List]]: Parameter grids for each model
    """
    return {
        'Ridge Regression': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'Lasso Regression': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'Support Vector Regression': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        }
    }