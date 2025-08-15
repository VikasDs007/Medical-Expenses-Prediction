# 🏥 AI Insurance Cost Predictor

A professional machine learning application that predicts medical insurance costs using advanced algorithms and provides actionable insights through an intuitive web interface.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project demonstrates a complete machine learning pipeline for predicting medical insurance costs. It features multiple ML models, comprehensive data analysis, and a professional web application that provides practical, actionable insights.

### 🌟 Key Features
- **🤖 Multiple ML Models**: Random Forest, Gradient Boosting, Linear Regression, and SVR
- **💡 Smart Insights**: Personalized recommendations with quantified savings potential
- **📊 Visual Analytics**: Interactive charts showing cost breakdowns and comparisons
- **🎯 BMI Calculator**: Integrated health assessment with color-coded categories
- **📈 Population Comparison**: See how you compare to others in your demographic
- **💾 Export Functionality**: Download predictions as JSON or CSV reports
- **📱 Responsive Design**: Professional interface that works on all devices

## 🚀 Quick Start

### Option 1: One-Click Run
```bash
# Clone and run immediately
git clone https://github.com/yourusername/ai-insurance-predictor.git
cd ai-insurance-predictor
python run_app.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/main.py
```

## 📊 Dataset Information

The dataset contains **1,338 insurance records** with these features:

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| **Age** | Primary beneficiary age | Numeric | 18-64 years |
| **Sex** | Gender | Categorical | male, female |
| **BMI** | Body Mass Index | Numeric | 15.96-53.13 |
| **Children** | Number of dependents | Numeric | 0-5 |
| **Smoker** | Smoking status | Categorical | yes, no |
| **Region** | US residential area | Categorical | northeast, northwest, southeast, southwest |
| **Charges** | Annual medical costs | Numeric | $1,122-$63,770 |

## 🏗️ Project Structure

```
ai-insurance-predictor/
├── app/
│   ├── main.py                 # Main Streamlit application
│   └── utils/                  # Utility functions
│       ├── data_loader.py      # Data loading
│       ├── preprocessing.py    # Data preprocessing
│       └── model_utils.py      # Model utilities
├── data/
│   ├── raw/insurance.csv       # Original dataset
│   └── insurance.csv           # Working copy
├── models/                     # Trained ML models
│   ├── gradient_boosting_model.pkl
│   ├── random_forest_model.pkl
│   ├── linear_regression_model.pkl
│   └── *.pkl                   # Encoders and other models
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/                        # Source code modules
├── tests/                      # Unit tests
├── requirements.txt            # Dependencies
├── train_models.py            # Model training script
└── README.md                  # This file
```

## 🤖 Model Performance

| Model | R² Score | RMSE | MAE | Best For |
|-------|----------|------|-----|----------|
| **Gradient Boosting** 🏆 | **0.884** | **$4,389** | **$2,634** | **Overall accuracy** |
| Random Forest | 0.871 | $4,632 | $2,851 | Feature importance |
| Linear Regression | 0.783 | $6,062 | $4,185 | Interpretability |
| SVR | 0.756 | $6,891 | $4,892 | Non-linear patterns |

## 💡 Key Insights from Analysis

### 🚬 Smoking Impact
- **400%+ cost increase** for smokers vs non-smokers
- Average smoker cost: **$32,050**
- Average non-smoker cost: **$8,434**
- **Potential savings: $23,616/year** by quitting

### 📈 Age Factor
- **$257 average increase** per year of age
- Costs accelerate significantly after age 50
- Preventive care becomes crucial for cost management

### ⚖️ BMI Correlation
- **Strong correlation** between BMI and costs
- Obesity (BMI > 30) adds significant premium
- **Weight management** can provide substantial savings

### 🗺️ Regional Variations
- **Southeast**: Highest average costs ($14,735)
- **Southwest**: Moderate costs ($12,347)
- **Northwest**: Lowest average costs ($12,417)

## 🎨 Application Features

### 🎯 Smart Prediction Engine
- **Multiple ML models** with automatic best-model selection
- **Confidence intervals** for prediction reliability
- **Population percentile** ranking
- **Risk categorization** (Low/Medium/High cost)

### 💡 Actionable Insights
- **Peer comparison** with similar demographic profiles
- **Quantified savings** from lifestyle improvements
- **Specific recommendations** with action plans
- **Timeline guidance** for health improvements

### 📊 Visual Analytics
- **Cost factor breakdown** pie charts
- **Population comparison** charts by age group
- **Feature importance** visualization
- **BMI health category** color coding

### 💾 Professional Reporting
- **JSON export** with complete prediction details
- **CSV summary** for spreadsheet analysis
- **Timestamped reports** for tracking over time
- **Professional formatting** ready for documentation

## 🔧 Technical Implementation

### Machine Learning Pipeline
```python
# Model training and evaluation
models = {
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Linear Regression': LinearRegression(),
    'SVR': SVR(kernel='rbf', C=1000, gamma=0.1)
}
```

### Data Processing
- **Label encoding** for categorical variables
- **Feature scaling** for optimal model performance
- **Train/test split** with stratification
- **Cross-validation** for robust evaluation

### Web Application
- **Streamlit framework** for rapid development
- **Plotly visualizations** for interactive charts
- **Responsive CSS** for professional appearance
- **Session state management** for user experience

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app/main.py
```

### Streamlit Cloud (Recommended)
1. Push to GitHub repository
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py"]
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Test coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/ai-insurance-predictor.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- 💼 LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- 🐙 GitHub: [Your GitHub Profile](https://github.com/yourusername)
- 📧 Email: your.email@example.com

## 🙏 Acknowledgments

- **Dataset**: [Medical Cost Personal Datasets](https://www.kaggle.com/mirichoi0218/insurance) from Kaggle
- **Libraries**: Streamlit, Scikit-learn, Plotly, Pandas, NumPy
- **Inspiration**: Healthcare cost transparency and accessibility

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ for the data science community

</div>