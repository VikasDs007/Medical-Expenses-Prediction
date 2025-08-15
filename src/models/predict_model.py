"""
Prediction functions for the medical insurance prediction project.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsurancePredictor:
    """
    A class to handle predictions for insurance cost estimation.
    """
    
    def __init__(self, model_path: Optional[str] = None, preprocessor_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path (str, optional): Path to the trained model
            preprocessor_path (str, optional): Path to the fitted preprocessor
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
        
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            self.is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise
    
    def load_preprocessor(self, preprocessor_path: str) -> None:
        """
        Load a fitted preprocessor from disk.
        
        Args:
            preprocessor_path (str): Path to the saved preprocessor
        """
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        except Exception as e:
            logger.error(f"Failed to load preprocessor from {preprocessor_path}: {str(e)}")
            raise
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data (Dict[str, Any]): Raw input data
            
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        if self.preprocessor:
            # Use the fitted preprocessor
            processed_data = self.preprocessor.transform(df)
        else:
            # Manual preprocessing (fallback)
            processed_data = self._manual_preprocessing(df)
        
        return processed_data
    
    def _manual_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Manual preprocessing when no preprocessor is available.
        
        Args:
            data (pd.DataFrame): Raw input data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        processed_data = data.copy()
        
        # Encode categorical variables (using simple mapping)
        sex_mapping = {'male': 1, 'female': 0}
        smoker_mapping = {'yes': 1, 'no': 0}
        region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
        
        if 'sex' in processed_data.columns:
            processed_data['sex_encoded'] = processed_data['sex'].map(sex_mapping)
        
        if 'smoker' in processed_data.columns:
            processed_data['smoker_encoded'] = processed_data['smoker'].map(smoker_mapping)
        
        if 'region' in processed_data.columns:
            processed_data['region_encoded'] = processed_data['region'].map(region_mapping)
        
        # Feature engineering
        processed_data['age_bmi_interaction'] = processed_data['age'] * processed_data['bmi']
        
        if 'smoker_encoded' in processed_data.columns:
            processed_data['smoker_bmi_interaction'] = (
                processed_data['smoker_encoded'] * processed_data['bmi']
            )
            processed_data['smoker_age_interaction'] = (
                processed_data['smoker_encoded'] * processed_data['age']
            )
        
        # Polynomial features
        processed_data['age_squared'] = processed_data['age'] ** 2
        processed_data['bmi_squared'] = processed_data['bmi'] ** 2
        
        # Age groups
        age_group_mapping = {
            (0, 25): 0,    # Young
            (25, 35): 1,   # Adult
            (35, 50): 2,   # Middle-aged
            (50, 100): 3   # Senior
        }
        
        def get_age_group(age):
            for (min_age, max_age), group in age_group_mapping.items():
                if min_age <= age < max_age:
                    return group
            return 3  # Default to Senior
        
        processed_data['age_group_encoded'] = processed_data['age'].apply(get_age_group)
        
        # BMI categories
        bmi_category_mapping = {
            (0, 18.5): 0,    # Underweight
            (18.5, 25): 1,   # Normal
            (25, 30): 2,     # Overweight
            (30, 100): 3     # Obese
        }
        
        def get_bmi_category(bmi):
            for (min_bmi, max_bmi), category in bmi_category_mapping.items():
                if min_bmi <= bmi < max_bmi:
                    return category
            return 3  # Default to Obese
        
        processed_data['bmi_category_encoded'] = processed_data['bmi'].apply(get_bmi_category)
        
        # Family size
        processed_data['family_size'] = processed_data['children'] + 1
        
        # Select relevant features
        feature_columns = [
            'age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded',
            'age_bmi_interaction', 'smoker_bmi_interaction', 'smoker_age_interaction',
            'age_squared', 'bmi_squared', 'age_group_encoded', 'bmi_category_encoded',
            'family_size'
        ]
        
        # Keep only available features
        available_features = [col for col in feature_columns if col in processed_data.columns]
        
        return processed_data[available_features]
    
    def predict(self, input_data: Dict[str, Any]) -> float:
        """
        Make a single prediction.
        
        Args:
            input_data (Dict[str, Any]): Input data for prediction
            
        Returns:
            float: Predicted insurance cost
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Preprocess the input
        processed_data = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)[0]
        
        logger.info(f"Prediction made: ${prediction:.2f}")
        return prediction
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]]) -> List[float]:
        """
        Make predictions for multiple inputs.
        
        Args:
            input_data_list (List[Dict[str, Any]]): List of input data
            
        Returns:
            List[float]: List of predicted insurance costs
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        predictions = []
        
        for input_data in input_data_list:
            try:
                prediction = self.predict(input_data)
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Failed to predict for input {input_data}: {str(e)}")
                predictions.append(None)
        
        return predictions
    
    def predict_with_confidence(self, input_data: Dict[str, Any], 
                               confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """
        Make a prediction with confidence intervals (for ensemble models).
        
        Args:
            input_data (Dict[str, Any]): Input data for prediction
            confidence_level (float): Confidence level for intervals
            
        Returns:
            Tuple[float, float, float]: (prediction, lower_bound, upper_bound)
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Preprocess the input
        processed_data = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)[0]
        
        # Calculate confidence intervals (if possible)
        if hasattr(self.model, 'estimators_'):
            # For ensemble methods
            individual_predictions = [
                estimator.predict(processed_data)[0] 
                for estimator in self.model.estimators_
            ]
            
            std_pred = np.std(individual_predictions)
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
            
            lower_bound = prediction - z_score * std_pred
            upper_bound = prediction + z_score * std_pred
        else:
            # For other models, use a simple heuristic
            error_margin = prediction * 0.1  # 10% margin
            lower_bound = prediction - error_margin
            upper_bound = prediction + error_margin
        
        return prediction, lower_bound, upper_bound
    
    def explain_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide explanation for a prediction.
        
        Args:
            input_data (Dict[str, Any]): Input data for prediction
            
        Returns:
            Dict[str, Any]: Explanation of the prediction
        """
        prediction = self.predict(input_data)
        
        explanation = {
            'prediction': prediction,
            'input_data': input_data,
            'factors': []
        }
        
        # Analyze key factors
        if input_data.get('smoker') == 'yes':
            explanation['factors'].append({
                'factor': 'Smoking Status',
                'impact': 'High',
                'description': 'Smoking significantly increases insurance costs'
            })
        
        age = input_data.get('age', 0)
        if age > 50:
            explanation['factors'].append({
                'factor': 'Age',
                'impact': 'Medium',
                'description': f'Age {age} is associated with higher medical costs'
            })
        
        bmi = input_data.get('bmi', 0)
        if bmi > 30:
            explanation['factors'].append({
                'factor': 'BMI',
                'impact': 'Medium',
                'description': f'BMI {bmi:.1f} indicates obesity, which may increase costs'
            })
        
        children = input_data.get('children', 0)
        if children > 2:
            explanation['factors'].append({
                'factor': 'Number of Children',
                'impact': 'Low',
                'description': f'Having {children} children may slightly affect costs'
            })
        
        return explanation
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the loaded model.
        
        Returns:
            pd.DataFrame: Feature importance data or None if not available
        """
        if not self.is_loaded or self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            return None
        
        # Create feature names if not available
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

def create_prediction_report(predictor: InsurancePredictor, 
                           input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive prediction report.
    
    Args:
        predictor (InsurancePredictor): Fitted predictor
        input_data (Dict[str, Any]): Input data for prediction
        
    Returns:
        Dict[str, Any]: Comprehensive prediction report
    """
    # Make prediction with confidence
    prediction, lower_bound, upper_bound = predictor.predict_with_confidence(input_data)
    
    # Get explanation
    explanation = predictor.explain_prediction(input_data)
    
    # Create report
    report = {
        'prediction': {
            'estimated_cost': prediction,
            'confidence_interval': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        },
        'input_summary': input_data,
        'risk_factors': explanation['factors'],
        'recommendations': []
    }
    
    # Add recommendations based on risk factors
    if input_data.get('smoker') == 'yes':
        report['recommendations'].append(
            "Consider smoking cessation programs to reduce health risks and insurance costs"
        )
    
    if input_data.get('bmi', 0) > 30:
        report['recommendations'].append(
            "Maintaining a healthy weight through diet and exercise may help reduce costs"
        )
    
    if input_data.get('age', 0) > 50:
        report['recommendations'].append(
            "Regular health check-ups become increasingly important with age"
        )
    
    return report