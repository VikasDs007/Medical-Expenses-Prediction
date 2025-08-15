import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional
import os

@st.cache_data
def load_insurance_data(file_path: str = 'data/raw/insurance.csv') -> Optional[pd.DataFrame]:
    """
    Load the insurance dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset or None if file not found
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"Dataset not found at {file_path}. Please ensure the file exists.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def get_data_info(data: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        data (pd.DataFrame): The dataset
        
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns),
        'summary_stats': data.describe().to_dict()
    }
    return info

def validate_data(data: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate the insurance dataset for required columns and data quality.
    
    Args:
        data (pd.DataFrame): The dataset to validate
        
    Returns:
        Tuple[bool, list]: (is_valid, list_of_issues)
    """
    required_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    issues = []
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    
    # Check for empty dataset
    if len(data) == 0:
        issues.append("Dataset is empty")
    
    # Check for missing values in critical columns
    if 'charges' in data.columns and data['charges'].isnull().sum() > 0:
        issues.append("Target variable 'charges' has missing values")
    
    # Check data types
    if 'age' in data.columns and not pd.api.types.is_numeric_dtype(data['age']):
        issues.append("'age' column should be numeric")
    
    if 'bmi' in data.columns and not pd.api.types.is_numeric_dtype(data['bmi']):
        issues.append("'bmi' column should be numeric")
    
    if 'children' in data.columns and not pd.api.types.is_numeric_dtype(data['children']):
        issues.append("'children' column should be numeric")
    
    if 'charges' in data.columns and not pd.api.types.is_numeric_dtype(data['charges']):
        issues.append("'charges' column should be numeric")
    
    # Check for reasonable value ranges
    if 'age' in data.columns:
        if data['age'].min() < 0 or data['age'].max() > 150:
            issues.append("Age values seem unreasonable (should be between 0-150)")
    
    if 'bmi' in data.columns:
        if data['bmi'].min() < 10 or data['bmi'].max() > 60:
            issues.append("BMI values seem unreasonable (should be between 10-60)")
    
    if 'charges' in data.columns:
        if data['charges'].min() < 0:
            issues.append("Insurance charges cannot be negative")
    
    is_valid = len(issues) == 0
    return is_valid, issues