#!/usr/bin/env python3
"""
Demo script to show sample predictions from the trained models.
"""

import pandas as pd
import joblib
import numpy as np
from datetime import datetime

def load_models_and_encoders():
    """Load trained models and encoders."""
    try:
        models = {
            'Linear Regression': joblib.load('models/linear_regression.pkl'),
            'Random Forest': joblib.load('models/random_forest.pkl'),
            'Gradient Boosting': joblib.load('models/gradient_boosting.pkl')
        }
        
        encoders = {
            'sex': joblib.load('models/le_sex.pkl'),
            'smoker': joblib.load('models/le_smoker.pkl'),
            'region': joblib.load('models/le_region.pkl')
        }
        
        return models, encoders
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Please run 'python train_models.py' first to train the models.")
        return None, None

def preprocess_input(input_data, encoders):
    """Preprocess input data for prediction."""
    processed = input_data.copy()
    
    # Encode categorical variables
    processed['sex_encoded'] = encoders['sex'].transform(processed['sex'])
    processed['smoker_encoded'] = encoders['smoker'].transform(processed['smoker'])
    processed['region_encoded'] = encoders['region'].transform(processed['region'])
    
    # Feature engineering
    processed['age_bmi_interaction'] = processed['age'] * processed['bmi']
    processed['smoker_bmi_interaction'] = processed['smoker_encoded'] * processed['bmi']
    processed['age_squared'] = processed['age'] ** 2
    processed['bmi_squared'] = processed['bmi'] ** 2
    
    # Select features
    features = ['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 
                'region_encoded', 'age_bmi_interaction', 'smoker_bmi_interaction',
                'age_squared', 'bmi_squared']
    
    return processed[features]

def make_prediction(models, encoders, person_data):
    """Make prediction for a person."""
    # Create DataFrame
    input_df = pd.DataFrame([person_data])
    
    # Preprocess
    X = preprocess_input(input_df, encoders)
    
    # Make predictions with all models
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(X)[0]
        predictions[model_name] = pred
    
    return predictions

def main():
    """Run demo predictions."""
    print("🏥 Medical Insurance Cost Prediction - Demo")
    print("=" * 50)
    
    # Load models
    models, encoders = load_models_and_encoders()
    if models is None:
        return
    
    # Sample profiles for demonstration
    sample_profiles = [
        {
            "name": "Young Professional",
            "age": 28,
            "sex": "male",
            "bmi": 24.5,
            "children": 0,
            "smoker": "no",
            "region": "northeast"
        },
        {
            "name": "Middle-aged Parent",
            "age": 42,
            "sex": "female",
            "bmi": 27.8,
            "children": 2,
            "smoker": "no",
            "region": "southeast"
        },
        {
            "name": "Senior Smoker",
            "age": 58,
            "sex": "male",
            "bmi": 29.2,
            "children": 0,
            "smoker": "yes",
            "region": "southwest"
        },
        {
            "name": "Young Smoker",
            "age": 25,
            "sex": "female",
            "bmi": 32.1,
            "children": 1,
            "smoker": "yes",
            "region": "northwest"
        }
    ]
    
    print(f"\nRunning predictions for {len(sample_profiles)} sample profiles...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n")
    
    results = []
    
    for profile in sample_profiles:
        name = profile.pop("name")
        predictions = make_prediction(models, encoders, profile)
        
        print(f"👤 {name}")
        print(f"   Profile: {profile['age']}yr {profile['sex']}, BMI {profile['bmi']}, "
              f"{profile['children']} children, {profile['smoker']} smoker, {profile['region']}")
        print("   Predictions:")
        
        for model_name, pred in predictions.items():
            print(f"   • {model_name:20}: ${pred:8,.2f}")
        
        # Find best and worst predictions
        best_pred = min(predictions.values())
        worst_pred = max(predictions.values())
        range_pred = worst_pred - best_pred
        
        print(f"   📊 Range: ${range_pred:,.2f} (${best_pred:,.2f} - ${worst_pred:,.2f})")
        print()
        
        # Store results
        result = {"name": name, **profile, **predictions}
        results.append(result)
    
    # Summary statistics
    print("📊 SUMMARY STATISTICS")
    print("-" * 30)
    
    results_df = pd.DataFrame(results)
    
    print(f"Average predictions:")
    for model in ['Linear Regression', 'Random Forest', 'Gradient Boosting']:
        avg_pred = results_df[model].mean()
        print(f"• {model:20}: ${avg_pred:8,.2f}")
    
    print(f"\nSmoking impact analysis:")
    smokers = results_df[results_df['smoker'] == 'yes']
    non_smokers = results_df[results_df['smoker'] == 'no']
    
    if len(smokers) > 0 and len(non_smokers) > 0:
        smoker_avg = smokers['Gradient Boosting'].mean()
        non_smoker_avg = non_smokers['Gradient Boosting'].mean()
        smoking_multiplier = smoker_avg / non_smoker_avg
        
        print(f"• Smokers average:     ${smoker_avg:8,.2f}")
        print(f"• Non-smokers average: ${non_smoker_avg:8,.2f}")
        print(f"• Smoking multiplier:  {smoking_multiplier:8.1f}x")
    
    print(f"\nAge impact analysis:")
    young = results_df[results_df['age'] < 35]
    older = results_df[results_df['age'] >= 35]
    
    if len(young) > 0 and len(older) > 0:
        young_avg = young['Gradient Boosting'].mean()
        older_avg = older['Gradient Boosting'].mean()
        age_multiplier = older_avg / young_avg
        
        print(f"• Under 35 average:    ${young_avg:8,.2f}")
        print(f"• 35+ average:         ${older_avg:8,.2f}")
        print(f"• Age multiplier:      {age_multiplier:8.1f}x")
    
    print("\n🎯 Key Insights:")
    print("• Smoking status has the largest impact on insurance costs")
    print("• Age and BMI create compound effects when combined")
    print("• Regional differences can account for $1,000+ variations")
    print("• All models generally agree on risk factors")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"💡 Try the interactive web app: streamlit run app/main.py")

if __name__ == "__main__":
    main()