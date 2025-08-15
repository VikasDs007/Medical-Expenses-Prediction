#!/usr/bin/env python3
"""
Script to train and save machine learning models for insurance cost prediction.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path='data/raw/insurance.csv'):
    """Load and preprocess the insurance data."""
    logger.info("Loading data...")
    data = pd.read_csv(file_path)
    
    logger.info(f"Data shape: {data.shape}")
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    data['sex_encoded'] = le_sex.fit_transform(data['sex'])
    data['smoker_encoded'] = le_smoker.fit_transform(data['smoker'])
    data['region_encoded'] = le_region.fit_transform(data['region'])
    
    # Feature engineering
    data['age_bmi_interaction'] = data['age'] * data['bmi']
    data['smoker_bmi_interaction'] = data['smoker_encoded'] * data['bmi']
    data['age_squared'] = data['age'] ** 2
    data['bmi_squared'] = data['bmi'] ** 2
    
    # Select features
    features = ['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 
                'region_encoded', 'age_bmi_interaction', 'smoker_bmi_interaction',
                'age_squared', 'bmi_squared']
    
    X = data[features]
    y = data['charges']
    
    # Save encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(le_sex, 'models/le_sex.pkl')
    joblib.dump(le_smoker, 'models/le_smoker.pkl')
    joblib.dump(le_region, 'models/le_region.pkl')
    
    logger.info("Data preprocessing completed")
    return X, y, data

def train_models(X, y):
    """Train multiple models and save them."""
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        # Save model
        model_path = f'models/{name}.pkl'
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        logger.info(f"{name} - MAE: ${mae:,.0f}, RMSE: ${rmse:,.0f}, R²: {r2:.3f}")
    
    return results, X_test, y_test

def save_results(results, X_test, y_test):
    """Save training results and test data."""
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('models/training_results.csv')
    
    # Save test data for later evaluation
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('models/test_data.csv', index=False)
    
    logger.info("Results and test data saved")

def main():
    """Main training pipeline."""
    try:
        # Load and preprocess data
        X, y, data = load_and_preprocess_data()
        
        # Train models
        results, X_test, y_test = train_models(X, y)
        
        # Save results
        save_results(results, X_test, y_test)
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        
        print(f"Dataset size: {len(data):,} records")
        print(f"Features used: {len(X.columns)}")
        print(f"Test set size: {len(X_test):,} records")
        
        print("\nModel Performance:")
        print("-" * 30)
        for model_name, metrics in results.items():
            print(f"{model_name.replace('_', ' ').title()}:")
            print(f"  MAE: ${metrics['mae']:,.0f}")
            print(f"  RMSE: ${metrics['rmse']:,.0f}")
            print(f"  R²: {metrics['r2']:.3f}")
            print()
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['r2'])
        print(f"Best model: {best_model.replace('_', ' ').title()} (R² = {results[best_model]['r2']:.3f})")
        
        print("\nFiles created:")
        print("- models/linear_regression.pkl")
        print("- models/random_forest.pkl")
        print("- models/gradient_boosting.pkl")
        print("- models/le_sex.pkl")
        print("- models/le_smoker.pkl")
        print("- models/le_region.pkl")
        print("- models/training_results.csv")
        print("- models/test_data.csv")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()