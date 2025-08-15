"""
Tests for model training and prediction functions.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.models.train_model import ModelTrainer
from src.models.predict_model import InsurancePredictor

class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        
        self.X_train = pd.DataFrame({
            'age': np.random.randint(18, 65, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'children': np.random.randint(0, 4, n_samples),
            'sex_encoded': np.random.randint(0, 2, n_samples),
            'smoker_encoded': np.random.randint(0, 2, n_samples),
            'region_encoded': np.random.randint(0, 4, n_samples)
        })
        
        # Create synthetic target variable
        self.y_train = (
            self.X_train['age'] * 100 +
            self.X_train['bmi'] * 200 +
            self.X_train['smoker_encoded'] * 10000 +
            np.random.normal(0, 1000, n_samples)
        )
        
        # Create test data
        n_test = 20
        self.X_test = pd.DataFrame({
            'age': np.random.randint(18, 65, n_test),
            'bmi': np.random.normal(25, 5, n_test),
            'children': np.random.randint(0, 4, n_test),
            'sex_encoded': np.random.randint(0, 2, n_test),
            'smoker_encoded': np.random.randint(0, 2, n_test),
            'region_encoded': np.random.randint(0, 4, n_test)
        })
        
        self.y_test = (
            self.X_test['age'] * 100 +
            self.X_test['bmi'] * 200 +
            self.X_test['smoker_encoded'] * 10000 +
            np.random.normal(0, 1000, n_test)
        )
        
        self.trainer = ModelTrainer()
    
    def test_initialization(self):
        """Test ModelTrainer initialization."""
        self.assertIsInstance(self.trainer.models, dict)
        self.assertIn('Linear Regression', self.trainer.models)
        self.assertIn('Random Forest', self.trainer.models)
        self.assertEqual(len(self.trainer.trained_models), 0)
        self.assertEqual(len(self.trainer.results), 0)
    
    def test_train_single_model(self):
        """Test training a single model."""
        model = self.trainer.train_single_model('Linear Regression', self.X_train, self.y_train)
        
        self.assertIsInstance(model, LinearRegression)
        self.assertIn('Linear Regression', self.trainer.trained_models)
        self.assertTrue(hasattr(model, 'coef_'))
    
    def test_train_all_models(self):
        """Test training all models."""
        trained_models = self.trainer.train_all_models(self.X_train, self.y_train)
        
        self.assertIsInstance(trained_models, dict)
        self.assertGreater(len(trained_models), 0)
        self.assertIn('Linear Regression', trained_models)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Train a model first
        self.trainer.train_single_model('Linear Regression', self.X_train, self.y_train)
        
        # Evaluate the model
        metrics = self.trainer.evaluate_model('Linear Regression', self.X_test, self.y_test)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('mse', metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['r2'], -1)  # R² should be > -1
        self.assertGreater(metrics['mae'], 0)  # MAE should be positive
        self.assertGreater(metrics['rmse'], 0)  # RMSE should be positive
    
    def test_evaluate_all_models(self):
        """Test evaluation of all models."""
        # Train models first
        self.trainer.train_all_models(self.X_train, self.y_train)
        
        # Evaluate all models
        results = self.trainer.evaluate_all_models(self.X_test, self.y_test)
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that each result has the required metrics
        for model_name, metrics in results.items():
            self.assertIn('mae', metrics)
            self.assertIn('rmse', metrics)
            self.assertIn('r2', metrics)
    
    def test_get_best_model(self):
        """Test getting the best model."""
        # Train and evaluate models first
        self.trainer.train_all_models(self.X_train, self.y_train)
        self.trainer.evaluate_all_models(self.X_test, self.y_test)
        
        # Get best model
        best_name, best_model = self.trainer.get_best_model('r2')
        
        self.assertIsInstance(best_name, str)
        self.assertIsNotNone(best_model)
        self.assertIn(best_name, self.trainer.trained_models)
    
    def test_cross_validate_model(self):
        """Test cross-validation."""
        cv_results = self.trainer.cross_validate_model('Linear Regression', self.X_train, self.y_train, cv=3)
        
        self.assertIsInstance(cv_results, dict)
        self.assertIn('cv_r2_mean', cv_results)
        self.assertIn('cv_r2_std', cv_results)
        self.assertIn('cv_mae_mean', cv_results)
        self.assertIn('cv_rmse_mean', cv_results)
    
    def test_get_results_summary(self):
        """Test getting results summary."""
        # Train and evaluate models first
        self.trainer.train_all_models(self.X_train, self.y_train)
        self.trainer.evaluate_all_models(self.X_test, self.y_test)
        
        summary = self.trainer.get_results_summary()
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn('Model', summary.columns)
        self.assertIn('R²', summary.columns)
        self.assertIn('MAE', summary.columns)
        self.assertIn('RMSE', summary.columns)

class TestInsurancePredictor(unittest.TestCase):
    """Test cases for InsurancePredictor class."""
    
    def setUp(self):
        """Set up test data."""
        self.predictor = InsurancePredictor()
        
        # Create and train a simple model for testing
        np.random.seed(42)
        X_train = pd.DataFrame({
            'age': np.random.randint(18, 65, 100),
            'bmi': np.random.normal(25, 5, 100),
            'children': np.random.randint(0, 4, 100),
            'sex_encoded': np.random.randint(0, 2, 100),
            'smoker_encoded': np.random.randint(0, 2, 100),
            'region_encoded': np.random.randint(0, 4, 100)
        })
        
        y_train = (
            X_train['age'] * 100 +
            X_train['bmi'] * 200 +
            X_train['smoker_encoded'] * 10000 +
            np.random.normal(0, 1000, 100)
        )
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.predictor.model = self.model
        self.predictor.is_loaded = True
        
        self.sample_input = {
            'age': 30,
            'sex': 'male',
            'bmi': 25.0,
            'children': 1,
            'smoker': 'no',
            'region': 'northeast'
        }
    
    def test_initialization(self):
        """Test InsurancePredictor initialization."""
        predictor = InsurancePredictor()
        self.assertIsNone(predictor.model)
        self.assertIsNone(predictor.preprocessor)
        self.assertFalse(predictor.is_loaded)
    
    def test_predict(self):
        """Test making a prediction."""
        prediction = self.predictor.predict(self.sample_input)
        
        self.assertIsInstance(prediction, (int, float))
        self.assertGreater(prediction, 0)  # Insurance cost should be positive
    
    def test_predict_batch(self):
        """Test batch prediction."""
        input_list = [self.sample_input, self.sample_input.copy()]
        predictions = self.predictor.predict_batch(input_list)
        
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
    
    def test_predict_with_confidence(self):
        """Test prediction with confidence intervals."""
        prediction, lower, upper = self.predictor.predict_with_confidence(self.sample_input)
        
        self.assertIsInstance(prediction, (int, float))
        self.assertIsInstance(lower, (int, float))
        self.assertIsInstance(upper, (int, float))
        self.assertLessEqual(lower, prediction)
        self.assertLessEqual(prediction, upper)
    
    def test_explain_prediction(self):
        """Test prediction explanation."""
        explanation = self.predictor.explain_prediction(self.sample_input)
        
        self.assertIsInstance(explanation, dict)
        self.assertIn('prediction', explanation)
        self.assertIn('input_data', explanation)
        self.assertIn('factors', explanation)
        self.assertIsInstance(explanation['factors'], list)
    
    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        predictor = InsurancePredictor()
        
        with self.assertRaises(ValueError):
            predictor.predict(self.sample_input)

if __name__ == '__main__':
    unittest.main()