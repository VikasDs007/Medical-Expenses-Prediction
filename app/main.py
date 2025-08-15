import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🏥 AI Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .savings-box {
        background: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the insurance dataset"""
    try:
        data = pd.read_csv('data/raw/insurance.csv')
        return data
    except FileNotFoundError:
        try:
            data = pd.read_csv('data/insurance.csv')
            return data
        except FileNotFoundError:
            st.error("Dataset not found. Please ensure the insurance dataset is available.")
            return None

@st.cache_resource
def load_models():
    """Load trained models and encoders"""
    models = {}
    encoders = {}
    
    model_files = {
        'Random Forest': 'models/random_forest_model.pkl',
        'Gradient Boosting': 'models/gradient_boosting_model.pkl',
        'Linear Regression': 'models/linear_regression_model.pkl',
        'SVR': 'models/svr_model.pkl'
    }
    
    encoder_files = {
        'sex': 'models/sex_encoder.pkl',
        'smoker': 'models/smoker_encoder.pkl',
        'region': 'models/region_encoder.pkl'
    }
    
    # Load models
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    
    # Load encoders
    for name, path in encoder_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                encoders[name] = pickle.load(f)
    
    return models if models else None, encoders if encoders else None

def train_models_if_needed():
    """Train models if they don't exist"""
    if not os.path.exists('models/gradient_boosting_model.pkl'):
        with st.spinner("Training models for the first time..."):
            data = load_data()
            if data is not None:
                # Prepare data
                le_sex = LabelEncoder()
                le_smoker = LabelEncoder()
                le_region = LabelEncoder()
                
                data_encoded = data.copy()
                data_encoded['sex'] = le_sex.fit_transform(data['sex'])
                data_encoded['smoker'] = le_smoker.fit_transform(data['smoker'])
                data_encoded['region'] = le_region.fit_transform(data['region'])
                
                X = data_encoded.drop('charges', axis=1)
                y = data_encoded['charges']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train models
                models = {
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'Linear Regression': LinearRegression(),
                    'SVR': SVR(kernel='rbf', C=1000, gamma=0.1)
                }
                
                # Create models directory
                os.makedirs('models', exist_ok=True)
                
                # Train and save models
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    filename = name.lower().replace(' ', '_') + '_model.pkl'
                    with open(f'models/{filename}', 'wb') as f:
                        pickle.dump(model, f)
                
                # Save encoders
                with open('models/sex_encoder.pkl', 'wb') as f:
                    pickle.dump(le_sex, f)
                with open('models/smoker_encoder.pkl', 'wb') as f:
                    pickle.dump(le_smoker, f)
                with open('models/region_encoder.pkl', 'wb') as f:
                    pickle.dump(le_region, f)
                
                st.success("Models trained successfully!")

def calculate_bmi_category(bmi):
    """Calculate BMI category and color"""
    if bmi < 18.5:
        return "Underweight", "🔵", "#3498db"
    elif bmi < 25:
        return "Normal", "🟢", "#2ecc71"
    elif bmi < 30:
        return "Overweight", "🟡", "#f39c12"
    else:
        return "Obese", "🔴", "#e74c3c"

def generate_practical_insights(age, bmi, smoker, children, prediction, data):
    """Generate practical, actionable insights"""
    insights = []
    
    # Compare with similar profiles
    similar_profiles = data[
        (abs(data['age'] - age) <= 5) &
        (abs(data['bmi'] - bmi) <= 3) &
        (data['smoker'] == smoker)
    ]
    
    if len(similar_profiles) > 0:
        avg_similar = similar_profiles['charges'].mean()
        if prediction > avg_similar:
            insights.append({
                'type': 'comparison',
                'title': '👥 Peer Comparison',
                'message': f'Your predicted cost (${prediction:,.0f}) is ${prediction - avg_similar:,.0f} higher than similar profiles (${avg_similar:,.0f})',
                'action': 'Consider lifestyle improvements to reduce costs'
            })
        else:
            insights.append({
                'type': 'comparison',
                'title': '👥 Peer Comparison',
                'message': f'Great! Your predicted cost (${prediction:,.0f}) is ${avg_similar - prediction:,.0f} lower than similar profiles (${avg_similar:,.0f})',
                'action': 'Keep up your healthy lifestyle!'
            })
    
    # Smoking insights
    if smoker == 'yes':
        nonsmoker_prediction = prediction * 0.6  # Approximate reduction
        savings = prediction - nonsmoker_prediction
        insights.append({
            'type': 'smoking',
            'title': '🚭 Smoking Impact',
            'message': f'Quitting smoking could save you approximately ${savings:,.0f} per year',
            'action': 'Consider smoking cessation programs - the health and financial benefits are immediate'
        })
    
    # BMI insights
    if bmi > 25:
        target_bmi = 24
        weight_to_lose = ((bmi - target_bmi) / bmi) * 100
        potential_savings = (bmi - 25) * 200  # Rough estimate
        insights.append({
            'type': 'weight',
            'title': '⚖️ Weight Management',
            'message': f'Reducing BMI to 24 (about {weight_to_lose:.1f}% weight loss) could save ~${potential_savings:,.0f}/year',
            'action': 'Consult with healthcare provider for a safe weight management plan'
        })
    
    # Age-related insights
    if age > 45:
        insights.append({
            'type': 'age',
            'title': '📅 Age Considerations',
            'message': f'At {age}, preventive care becomes increasingly important for cost management',
            'action': 'Regular health screenings can help catch issues early and reduce long-term costs'
        })
    
    # Children insights
    if children > 0:
        insights.append({
            'type': 'family',
            'title': '👨‍👩‍👧‍👦 Family Planning',
            'message': f'With {children} children, consider family health plans and preventive care for all',
            'action': 'Family wellness programs often provide better value than individual plans'
        })
    
    return insights

def create_cost_breakdown_chart(age, bmi, smoker, children, region):
    """Create a visual breakdown of cost factors"""
    # Estimate factor contributions
    base_cost = 3000
    age_contribution = (age - 18) * 100
    bmi_contribution = max(0, (bmi - 25) * 200)
    smoking_contribution = 25000 if smoker == 'yes' else 0
    children_contribution = children * 500
    regional_factors = {
        'northeast': 0, 'northwest': -300, 
        'southeast': 800, 'southwest': 200
    }
    regional_contribution = regional_factors.get(region, 0)
    
    # Create data for chart
    factors = ['Base Cost', 'Age Factor', 'BMI Factor', 'Smoking', 'Children', 'Regional']
    contributions = [
        base_cost, age_contribution, bmi_contribution, 
        smoking_contribution, children_contribution, abs(regional_contribution)
    ]
    
    # Filter out zero contributions for cleaner chart
    non_zero_factors = []
    non_zero_contributions = []
    for factor, contrib in zip(factors, contributions):
        if contrib > 0:
            non_zero_factors.append(factor)
            non_zero_contributions.append(contrib)
    
    # Create pie chart
    fig = px.pie(
        values=non_zero_contributions,
        names=non_zero_factors,
        title="Estimated Cost Factor Breakdown"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_comparison_chart(prediction, data, age, smoker):
    """Create comparison with population"""
    # Age groups
    age_groups = ['18-30', '31-45', '46-60', '60+']
    age_group = '18-30' if age <= 30 else '31-45' if age <= 45 else '46-60' if age <= 60 else '60+'
    
    # Calculate averages by age group and smoking status
    comparisons = []
    
    for group, (min_age, max_age) in [('18-30', (18, 30)), ('31-45', (31, 45)), ('46-60', (46, 60)), ('60+', (61, 100))]:
        group_data = data[(data['age'] >= min_age) & (data['age'] <= max_age)]
        
        smoker_avg = group_data[group_data['smoker'] == 'yes']['charges'].mean() if len(group_data[group_data['smoker'] == 'yes']) > 0 else 0
        nonsmoker_avg = group_data[group_data['smoker'] == 'no']['charges'].mean() if len(group_data[group_data['smoker'] == 'no']) > 0 else 0
        
        comparisons.append({
            'Age Group': group,
            'Smokers': smoker_avg,
            'Non-Smokers': nonsmoker_avg
        })
    
    df_comp = pd.DataFrame(comparisons)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Smokers',
        x=df_comp['Age Group'],
        y=df_comp['Smokers'],
        marker_color='#e74c3c'
    ))
    
    fig.add_trace(go.Bar(
        name='Non-Smokers',
        x=df_comp['Age Group'],
        y=df_comp['Non-Smokers'],
        marker_color='#2ecc71'
    ))
    
    # Add user's prediction as a line
    fig.add_trace(go.Scatter(
        x=df_comp['Age Group'],
        y=[prediction] * len(df_comp),
        mode='lines+markers',
        name='Your Prediction',
        line=dict(color='#3498db', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Your Prediction vs Population Averages',
        xaxis_title='Age Group',
        yaxis_title='Annual Cost ($)',
        barmode='group'
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <h1 class="main-header">🏥 AI Insurance Cost Predictor</h1>
    """, unsafe_allow_html=True)
    st.markdown("### Smart, actionable insights powered by machine learning")
    
    # Load data and models
    data = load_data()
    if data is None:
        st.stop()
    
    train_models_if_needed()
    models, encoders = load_models()
    
    if models is None or encoders is None:
        st.error("Could not load models. Please check if models are trained.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Settings")
        
        # Model selection
        selected_model = st.selectbox(
            "Choose Model:",
            list(models.keys()),
            index=1
        )
        
        # Display options
        st.markdown("### 📊 Display Options")
        show_insights = st.checkbox("Show Insights", value=True)
        show_breakdown = st.checkbox("Show Cost Breakdown", value=True)
        show_comparison = st.checkbox("Show Population Comparison", value=True)
        
        # Dataset info
        st.markdown("### 📈 Dataset Info")
        st.metric("Records", len(data))
        st.metric("Avg Cost", f"${data['charges'].mean():,.0f}")
        
        smoker_rate = (data['smoker'] == 'yes').mean() * 100
        st.metric("Smoker Rate", f"{smoker_rate:.1f}%")
    
    # Main input form
    st.markdown("## 📝 Your Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Personal Details")
        age = st.slider("Age", 18, 100, 32)
        sex = st.selectbox("Sex", ["male", "female"])
        children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
    
    with col2:
        st.markdown("### Health Information")
        
        # BMI Calculator
        st.markdown("**BMI Calculator**")
        height_cm = st.number_input("Height (cm)", 140, 220, 170)
        weight_kg = st.number_input("Weight (kg)", 40, 200, 70)
        
        bmi = weight_kg / ((height_cm / 100) ** 2)
        category, emoji, color = calculate_bmi_category(bmi)
        
        st.markdown(f"**Your BMI:** {bmi:.1f} {emoji} ({category})")
        
        smoker = st.selectbox("Smoking Status", ["no", "yes"])
    
    with col3:
        st.markdown("### Location")
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        
        # Quick stats for this region
        region_data = data[data['region'] == region]
        region_avg = region_data['charges'].mean()
        st.metric("Regional Average", f"${region_avg:,.0f}")
    
    # Prediction button
    if st.button("🔮 Get Prediction", type="primary"):
        # Prepare input
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [encoders['sex'].transform([sex])[0]],
            'bmi': [bmi],
            'children': [children],
            'smoker': [encoders['smoker'].transform([smoker])[0]],
            'region': [encoders['region'].transform([region])[0]]
        })
        
        # Make prediction
        model = models[selected_model]
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.markdown("---")
        st.markdown("## 🎯 Your Prediction")
        
        col_pred1, col_pred2, col_pred3 = st.columns([2, 1, 1])
        
        with col_pred1:
            st.markdown(f'<div class="prediction-box"><h2>${prediction:,.0f}</h2><p>Estimated Annual Cost</p></div>', unsafe_allow_html=True)
        
        with col_pred2:
            # Population percentile
            percentile = (data['charges'] < prediction).mean() * 100
            st.metric("Your Percentile", f"{percentile:.0f}%")
            
            # Comparison with average
            avg_cost = data['charges'].mean()
            diff = prediction - avg_cost
            if diff > 0:
                st.metric("Above Average", f"${diff:,.0f}")
            else:
                st.metric("Below Average", f"${abs(diff):,.0f}")
        
        with col_pred3:
            st.metric("Model Used", selected_model)
            
            # Risk category based on cost
            if prediction < 5000:
                risk = "Low Cost 🟢"
            elif prediction < 15000:
                risk = "Moderate Cost 🟡"
            else:
                risk = "High Cost 🔴"
            
            st.metric("Cost Category", risk)
        
        # Practical Insights
        if show_insights:
            st.markdown("## 💡 Practical Insights")
            insights = generate_practical_insights(age, bmi, smoker, children, prediction, data)
            
            for insight in insights:
                if insight['type'] == 'smoking':
                    st.success(f"**{insight['title']}**\n\n{insight['message']}\n\n**Action:** {insight['action']}")
                elif insight['type'] == 'weight':
                    st.warning(f"**{insight['title']}**\n\n{insight['message']}\n\n**Action:** {insight['action']}")
                else:
                    st.info(f"**{insight['title']}**\n\n{insight['message']}\n\n**Action:** {insight['action']}")
        
        # Cost Breakdown
        if show_breakdown:
            st.markdown("## 📊 Cost Factor Analysis")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                breakdown_fig = create_cost_breakdown_chart(age, bmi, smoker, children, region)
                st.plotly_chart(breakdown_fig, use_container_width=True)
            
            with col_chart2:
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    feature_names = ['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region']
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig_imp = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Model Feature Importance"
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
        
        # Population Comparison
        if show_comparison:
            st.markdown("## 📈 Population Comparison")
            comparison_fig = create_comparison_chart(prediction, data, age, smoker)
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Summary statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Dataset Average", f"${data['charges'].mean():,.0f}")
            with col_stat2:
                smoker_avg = data[data['smoker'] == 'yes']['charges'].mean()
                st.metric("Smoker Average", f"${smoker_avg:,.0f}")
            with col_stat3:
                nonsmoker_avg = data[data['smoker'] == 'no']['charges'].mean()
                st.metric("Non-smoker Average", f"${nonsmoker_avg:,.0f}")
            with col_stat4:
                smoking_premium = smoker_avg - nonsmoker_avg
                st.metric("Smoking Premium", f"${smoking_premium:,.0f}")
        
        # Export option
        st.markdown("## 💾 Export Results")
        
        # Create summary report
        report = {
            "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_profile": {
                "age": age,
                "sex": sex,
                "bmi": round(bmi, 2),
                "bmi_category": category,
                "children": children,
                "smoker": smoker,
                "region": region
            },
            "prediction": {
                "annual_cost": round(prediction, 2),
                "model_used": selected_model,
                "percentile": round(percentile, 1),
                "vs_average": round(diff, 2)
            }
        }
        
        report_json = json.dumps(report, indent=2)
        
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            st.download_button(
                label="📄 Download Report (JSON)",
                data=report_json,
                file_name=f"insurance_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col_export2:
            # Create CSV summary
            csv_data = pd.DataFrame([{
                'Date': datetime.now().strftime("%Y-%m-%d"),
                'Age': age,
                'Sex': sex,
                'BMI': round(bmi, 2),
                'Children': children,
                'Smoker': smoker,
                'Region': region,
                'Predicted_Cost': round(prediction, 2),
                'Model': selected_model,
                'Percentile': round(percentile, 1)
            }])
            
            st.download_button(
                label="📊 Download CSV",
                data=csv_data.to_csv(index=False),
                file_name=f"prediction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>AI Insurance Cost Predictor</strong></p>
        <p>Built with Streamlit • Powered by Machine Learning • Focused on Actionable Insights</p>
        <p><em>This tool provides estimates for educational purposes. Consult with insurance professionals for actual quotes.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()