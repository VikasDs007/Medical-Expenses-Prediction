"""
Data processing functions for the medical insurance prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the insurance dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {file_path}")
        raise

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the insurance dataset by handling missing values and outliers.
    
    Args:
        data (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    cleaned_data = data.copy()
    
    # Check for missing values
    missing_values = cleaned_data.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Found missing values: {missing_values[missing_values > 0].to_dict()}")
        # For this dataset, we'll drop rows with missing values
        cleaned_data = cleaned_data.dropna()
    
    # Remove duplicates
    initial_shape = cleaned_data.shape[0]
    cleaned_data = cleaned_data.drop_duplicates()
    if cleaned_data.shape[0] < initial_shape:
        logger.info(f"Removed {initial_shape - cleaned_data.shape[0]} duplicate rows")
    
    # Handle outliers in charges (using IQR method)
    Q1 = cleaned_data['charges'].quantile(0.25)
    Q3 = cleaned_data['charges'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = cleaned_data[(cleaned_data['charges'] < lower_bound) | 
                           (cleaned_data['charges'] > upper_bound)]
    
    if len(outliers) > 0:
        logger.info(f"Found {len(outliers)} outliers in charges column")
        # We'll keep outliers as they might be legitimate high-cost cases
    
    # Validate data ranges
    if (cleaned_data['age'] < 0).any() or (cleaned_data['age'] > 150).any():
        logger.warning("Found unrealistic age values")
    
    if (cleaned_data['bmi'] < 10).any() or (cleaned_data['bmi'] > 60).any():
        logger.warning("Found unrealistic BMI values")
    
    if (cleaned_data['charges'] < 0).any():
        logger.warning("Found negative charges")
        cleaned_data = cleaned_data[cleaned_data['charges'] >= 0]
    
    logger.info(f"Data cleaning completed. Final shape: {cleaned_data.shape}")
    return cleaned_data

def encode_categorical_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using Label Encoding.
    
    Args:
        data (pd.DataFrame): Dataset with categorical features
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, LabelEncoder]]: Encoded data and encoders
    """
    encoded_data = data.copy()
    encoders = {}
    
    categorical_columns = ['sex', 'smoker', 'region']
    
    for col in categorical_columns:
        if col in encoded_data.columns:
            le = LabelEncoder()
            encoded_data[f'{col}_encoded'] = le.fit_transform(encoded_data[col])
            encoders[col] = le
            logger.info(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return encoded_data, encoders

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features through feature engineering.
    
    Args:
        data (pd.DataFrame): Dataset with encoded categorical features
        
    Returns:
        pd.DataFrame: Dataset with new features
    """
    featured_data = data.copy()
    
    # Interaction features
    featured_data['age_bmi_interaction'] = featured_data['age'] * featured_data['bmi']
    
    if 'smoker_encoded' in featured_data.columns:
        featured_data['smoker_bmi_interaction'] = (
            featured_data['smoker_encoded'] * featured_data['bmi']
        )
        featured_data['smoker_age_interaction'] = (
            featured_data['smoker_encoded'] * featured_data['age']
        )
    
    # Polynomial features
    featured_data['age_squared'] = featured_data['age'] ** 2
    featured_data['bmi_squared'] = featured_data['bmi'] ** 2
    
    # Binned features
    featured_data['age_group'] = pd.cut(
        featured_data['age'],
        bins=[0, 25, 35, 50, 100],
        labels=['Young', 'Adult', 'Middle-aged', 'Senior']
    )
    
    featured_data['bmi_category'] = pd.cut(
        featured_data['bmi'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    
    # Encode the new categorical features
    if 'age_group' in featured_data.columns:
        featured_data['age_group_encoded'] = LabelEncoder().fit_transform(featured_data['age_group'])
    
    if 'bmi_category' in featured_data.columns:
        featured_data['bmi_category_encoded'] = LabelEncoder().fit_transform(featured_data['bmi_category'])
    
    # Family size feature
    featured_data['family_size'] = featured_data['children'] + 1  # +1 for the person themselves
    
    logger.info(f"Created {len(featured_data.columns) - len(data.columns)} new features")
    return featured_data

def prepare_features_target(data: pd.DataFrame, target_column: str = 'charges') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target variable for modeling.
    
    Args:
        data (pd.DataFrame): Processed dataset
        target_column (str): Name of the target column
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target variable
    """
    # Define feature columns (excluding original categorical columns and target)
    exclude_columns = ['sex', 'smoker', 'region', 'age_group', 'bmi_category', target_column]
    feature_columns = [col for col in data.columns if col not in exclude_columns]
    
    X = data[feature_columns]
    y = data[target_column]
    
    logger.info(f"Prepared {X.shape[1]} features for modeling")
    logger.info(f"Feature columns: {list(X.columns)}")
    
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    logger.info(f"Split data: Train shape {X_train.shape}, Test shape {X_test.shape}")
    return X_train, X_test, y_train, y_test

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        Tuple: Scaled X_train, scaled X_test, fitted scaler
    """
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    logger.info("Features scaled using StandardScaler")
    return X_train_scaled, X_test_scaled, scaler

def get_data_summary(data: pd.DataFrame) -> Dict:
    """
    Get a comprehensive summary of the dataset.
    
    Args:
        data (pd.DataFrame): Dataset to summarize
        
    Returns:
        Dict: Summary statistics and information
    """
    summary = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_summary': data.describe().to_dict(),
        'categorical_summary': {}
    }
    
    # Get categorical column summaries
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        summary['categorical_summary'][col] = data[col].value_counts().to_dict()
    
    return summary

def process_pipeline(file_path: str, target_column: str = 'charges') -> Dict:
    """
    Complete data processing pipeline.
    
    Args:
        file_path (str): Path to the raw data file
        target_column (str): Name of the target column
        
    Returns:
        Dict: Processed data and metadata
    """
    logger.info("Starting data processing pipeline")
    
    # Load and clean data
    raw_data = load_data(file_path)
    cleaned_data = clean_data(raw_data)
    
    # Encode categorical features
    encoded_data, encoders = encode_categorical_features(cleaned_data)
    
    # Create new features
    featured_data = create_features(encoded_data)
    
    # Prepare features and target
    X, y = prepare_features_target(featured_data, target_column)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    logger.info("Data processing pipeline completed")
    
    return {
        'raw_data': raw_data,
        'processed_data': featured_data,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }