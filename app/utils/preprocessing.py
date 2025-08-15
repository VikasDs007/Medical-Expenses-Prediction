import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import streamlit as st

class InsuranceDataPreprocessor:
    """
    A class to handle preprocessing of insurance data for machine learning models.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
    
    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            data (pd.DataFrame): Raw insurance data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Processed features and target variable
        """
        processed_data = data.copy()
        
        # Handle categorical variables
        categorical_columns = ['sex', 'smoker', 'region']
        
        for col in categorical_columns:
            if col in processed_data.columns:
                le = LabelEncoder()
                processed_data[f'{col}_encoded'] = le.fit_transform(processed_data[col])
                self.label_encoders[col] = le
        
        # Feature engineering
        processed_data['age_bmi_interaction'] = processed_data['age'] * processed_data['bmi']
        processed_data['smoker_bmi_interaction'] = (
            processed_data.get('smoker_encoded', 0) * processed_data['bmi']
        )
        processed_data['age_squared'] = processed_data['age'] ** 2
        processed_data['bmi_squared'] = processed_data['bmi'] ** 2
        
        # Create age groups
        processed_data['age_group'] = pd.cut(
            processed_data['age'], 
            bins=[0, 25, 35, 50, 100], 
            labels=['Young', 'Adult', 'Middle-aged', 'Senior']
        )
        processed_data['age_group_encoded'] = LabelEncoder().fit_transform(processed_data['age_group'])
        
        # Create BMI categories
        processed_data['bmi_category'] = pd.cut(
            processed_data['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        processed_data['bmi_category_encoded'] = LabelEncoder().fit_transform(processed_data['bmi_category'])
        
        # Select features for modeling
        self.feature_names = [
            'age', 'bmi', 'children',
            'sex_encoded', 'smoker_encoded', 'region_encoded',
            'age_bmi_interaction', 'smoker_bmi_interaction',
            'age_squared', 'bmi_squared',
            'age_group_encoded', 'bmi_category_encoded'
        ]
        
        # Filter features that exist in the data
        available_features = [f for f in self.feature_names if f in processed_data.columns]
        
        X = processed_data[available_features]
        y = processed_data['charges']
        
        self.is_fitted = True
        return X, y
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            data (pd.DataFrame): Raw insurance data
            
        Returns:
            pd.DataFrame: Processed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        processed_data = data.copy()
        
        # Apply label encoders
        for col, le in self.label_encoders.items():
            if col in processed_data.columns:
                processed_data[f'{col}_encoded'] = le.transform(processed_data[col])
        
        # Feature engineering (same as in fit_transform)
        processed_data['age_bmi_interaction'] = processed_data['age'] * processed_data['bmi']
        processed_data['smoker_bmi_interaction'] = (
            processed_data.get('smoker_encoded', 0) * processed_data['bmi']
        )
        processed_data['age_squared'] = processed_data['age'] ** 2
        processed_data['bmi_squared'] = processed_data['bmi'] ** 2
        
        # Age groups
        processed_data['age_group'] = pd.cut(
            processed_data['age'], 
            bins=[0, 25, 35, 50, 100], 
            labels=['Young', 'Adult', 'Middle-aged', 'Senior']
        )
        processed_data['age_group_encoded'] = LabelEncoder().fit_transform(processed_data['age_group'])
        
        # BMI categories
        processed_data['bmi_category'] = pd.cut(
            processed_data['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        processed_data['bmi_category_encoded'] = LabelEncoder().fit_transform(processed_data['bmi_category'])
        
        # Select features
        available_features = [f for f in self.feature_names if f in processed_data.columns]
        X = processed_data[available_features]
        
        return X

@st.cache_data
def prepare_data_for_modeling(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Prepare data for machine learning modeling.
    
    Args:
        data (pd.DataFrame): Raw insurance data
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        Dict[str, Any]: Dictionary containing processed data and preprocessor
    """
    preprocessor = InsuranceDataPreprocessor()
    X, y = preprocessor.fit_transform(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': X.columns.tolist()
    }

def get_feature_importance_data(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained scikit-learn model
        feature_names (list): List of feature names
        
    Returns:
        pd.DataFrame: Feature importance data
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df