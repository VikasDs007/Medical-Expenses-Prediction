"""
Tests for data processing functions.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from src.data.data_processing import (
    load_data, clean_data, encode_categorical_features,
    create_features, prepare_features_target, split_data
)

class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing functions."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'sex': ['male', 'female', 'male', 'female', 'male'],
            'bmi': [22.5, 25.0, 27.5, 30.0, 32.5],
            'children': [0, 1, 2, 1, 0],
            'smoker': ['no', 'yes', 'no', 'yes', 'no'],
            'region': ['northeast', 'southwest', 'southeast', 'northwest', 'northeast'],
            'charges': [1500.0, 15000.0, 3000.0, 12000.0, 2500.0]
        })
    
    def test_load_data(self):
        """Test data loading function."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test loading
            loaded_data = load_data(temp_file)
            self.assertIsInstance(loaded_data, pd.DataFrame)
            self.assertEqual(len(loaded_data), len(self.sample_data))
            self.assertEqual(list(loaded_data.columns), list(self.sample_data.columns))
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_load_data_file_not_found(self):
        """Test loading non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')
    
    def test_clean_data(self):
        """Test data cleaning function."""
        # Add some missing values and duplicates
        dirty_data = self.sample_data.copy()
        dirty_data.loc[0, 'age'] = np.nan
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[[0]]], ignore_index=True)
        
        cleaned_data = clean_data(dirty_data)
        
        # Check that missing values are handled
        self.assertFalse(cleaned_data.isnull().any().any())
        
        # Check that duplicates are removed
        self.assertEqual(len(cleaned_data), len(dirty_data.drop_duplicates().dropna()))
    
    def test_encode_categorical_features(self):
        """Test categorical encoding function."""
        encoded_data, encoders = encode_categorical_features(self.sample_data)
        
        # Check that encoded columns are created
        self.assertIn('sex_encoded', encoded_data.columns)
        self.assertIn('smoker_encoded', encoded_data.columns)
        self.assertIn('region_encoded', encoded_data.columns)
        
        # Check that encoders are returned
        self.assertIn('sex', encoders)
        self.assertIn('smoker', encoders)
        self.assertIn('region', encoders)
        
        # Check encoding values
        self.assertTrue(all(encoded_data['sex_encoded'].isin([0, 1])))
        self.assertTrue(all(encoded_data['smoker_encoded'].isin([0, 1])))
    
    def test_create_features(self):
        """Test feature creation function."""
        encoded_data, _ = encode_categorical_features(self.sample_data)
        featured_data = create_features(encoded_data)
        
        # Check that new features are created
        expected_features = [
            'age_bmi_interaction', 'smoker_bmi_interaction', 'smoker_age_interaction',
            'age_squared', 'bmi_squared', 'age_group_encoded', 'bmi_category_encoded',
            'family_size'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, featured_data.columns)
        
        # Check feature calculations
        self.assertTrue(all(
            featured_data['age_bmi_interaction'] == 
            featured_data['age'] * featured_data['bmi']
        ))
        
        self.assertTrue(all(
            featured_data['family_size'] == 
            featured_data['children'] + 1
        ))
    
    def test_prepare_features_target(self):
        """Test feature and target preparation."""
        encoded_data, _ = encode_categorical_features(self.sample_data)
        featured_data = create_features(encoded_data)
        
        X, y = prepare_features_target(featured_data)
        
        # Check that target is correct
        self.assertTrue(all(y == self.sample_data['charges']))
        
        # Check that original categorical columns are excluded
        self.assertNotIn('sex', X.columns)
        self.assertNotIn('smoker', X.columns)
        self.assertNotIn('region', X.columns)
        self.assertNotIn('charges', X.columns)
        
        # Check that encoded features are included
        self.assertIn('sex_encoded', X.columns)
        self.assertIn('smoker_encoded', X.columns)
    
    def test_split_data(self):
        """Test data splitting function."""
        encoded_data, _ = encode_categorical_features(self.sample_data)
        featured_data = create_features(encoded_data)
        X, y = prepare_features_target(featured_data)
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.4, random_state=42)
        
        # Check split proportions
        total_samples = len(X)
        train_samples = len(X_train)
        test_samples = len(X_test)
        
        self.assertEqual(train_samples + test_samples, total_samples)
        self.assertAlmostEqual(test_samples / total_samples, 0.4, places=1)
        
        # Check that indices match
        self.assertTrue(all(y_train.index == X_train.index))
        self.assertTrue(all(y_test.index == X_test.index))

if __name__ == '__main__':
    unittest.main()