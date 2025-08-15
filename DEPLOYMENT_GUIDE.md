# 🚀 Deployment Guide - AI Insurance Cost Predictor

## 📋 Overview
This guide covers multiple deployment options for your AI Insurance Cost Predictor application.

## 🏠 Local Development

### Prerequisites
```bash
pip install streamlit pandas numpy plotly seaborn matplotlib scikit-learn
```

### Running Locally
```bash
# Option 1: Simple version
streamlit run app/simple_main.py --server.port 8501

# Option 2: Enhanced version
streamlit run app/clean_main.py --server.port 8511

# Option 3: Final polished version
streamlit run app/final_app.py --server.port 8512
```

## ☁️ Cloud Deployment Options

### 1. Streamlit Cloud (Recommended)
**Pros:** Free, easy setup, automatic deployments
**Steps:**
1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository and main app file
5. Deploy with one click

**Requirements file (requirements.txt):**
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
seaborn>=0.12.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

### 2. Heroku Deployment
**Setup files needed:**

**Procfile:**
```
web: sh setup.sh && streamlit run app/final_app.py --server.port=$PORT --server.address=0.0.0.0
```

**setup.sh:**
```bash
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

**runtime.txt:**
```
python-3.9.18
```

### 3. Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/final_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run:**
```bash
docker build -t insurance-predictor .
docker run -p 8501:8501 insurance-predictor
```

### 4. AWS EC2 Deployment

**User Data Script:**
```bash
#!/bin/bash
yum update -y
yum install -y python3 python3-pip git

# Clone repository
git clone https://github.com/yourusername/Medical-Expenses-Prediction.git
cd Medical-Expenses-Prediction

# Install dependencies
pip3 install -r requirements.txt

# Run application
nohup streamlit run app/final_app.py --server.port 8501 --server.address 0.0.0.0 &
```

### 5. Google Cloud Platform (Cloud Run)

**cloudbuild.yaml:**
```yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/insurance-predictor', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/insurance-predictor']
- name: 'gcr.io/cloud-builders/gcloud'
  args: ['run', 'deploy', 'insurance-predictor', '--image', 'gcr.io/$PROJECT_ID/insurance-predictor', '--platform', 'managed', '--region', 'us-central1', '--allow-unauthenticated']
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Optional configurations
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_THEME_BASE=light
export STREAMLIT_THEME_PRIMARY_COLOR=#1f77b4
```

### Streamlit Config (.streamlit/config.toml)
```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false

[theme]
base = "light"
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[browser]
gatherUsageStats = false
```

## 📊 Performance Optimization

### Caching Strategy
- `@st.cache_data` for data loading
- `@st.cache_resource` for model loading
- Persistent model storage

### Memory Management
```python
# Clear cache when needed
st.cache_data.clear()
st.cache_resource.clear()
```

### Load Balancing (Production)
```nginx
upstream streamlit {
    server 127.0.0.1:8501;
    server 127.0.0.1:8502;
    server 127.0.0.1:8503;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://streamlit;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🔒 Security Considerations

### Authentication (Optional)
```python
import streamlit_authenticator as stauth

# Add to your app
authenticator = stauth.Authenticate(
    credentials,
    'some_cookie_name',
    'some_signature_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Your app code here
    pass
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
```

### HTTPS Setup
```bash
# Using Let's Encrypt with Certbot
sudo certbot --nginx -d your-domain.com
```

## 📈 Monitoring & Analytics

### Health Check Endpoint
```python
# Add to your app
@st.cache_data
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

# Access via: http://your-app.com/_stcore/health
```

### Usage Analytics
```python
# Google Analytics integration
st.components.v1.html("""
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
""", height=0)
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port
   netstat -tulpn | grep :8501
   # Kill process
   kill -9 <PID>
   ```

2. **Memory Issues**
   ```python
   # Reduce model size
   import joblib
   joblib.dump(model, 'model.pkl', compress=3)
   ```

3. **Slow Loading**
   ```python
   # Optimize data loading
   @st.cache_data(ttl=3600)  # Cache for 1 hour
   def load_data():
       return pd.read_csv('data.csv')
   ```

## 📱 Mobile Optimization

### Responsive Design
```css
/* Add to your CSS */
@media (max-width: 768px) {
    .main-header {
        font-size: 2rem !important;
    }
    
    .stColumns > div {
        min-width: 100% !important;
    }
}
```

## 🔄 CI/CD Pipeline

### GitHub Actions (.github/workflows/deploy.yml)
```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/
    
    - name: Deploy to Streamlit Cloud
      # Streamlit Cloud auto-deploys on push to main
      run: echo "Deployed to Streamlit Cloud"
```

## 📊 Scaling Considerations

### Horizontal Scaling
- Use load balancer (Nginx, HAProxy)
- Multiple Streamlit instances
- Shared model storage (Redis, S3)

### Vertical Scaling
- Increase server resources
- Optimize model size
- Use model quantization

## 🎯 Production Checklist

- [ ] Environment variables configured
- [ ] HTTPS enabled
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Health checks working
- [ ] Backup strategy in place
- [ ] Monitoring setup
- [ ] Performance optimized
- [ ] Security measures implemented
- [ ] Documentation updated

## 📞 Support

For deployment issues:
1. Check logs: `streamlit logs`
2. Verify requirements: `pip list`
3. Test locally first
4. Check firewall/security groups
5. Monitor resource usage

---

**🚀 Your app is now ready for production deployment!**