# 🚀 TrustCheckAI — Bias and Compliance Detaction Platform

<p align="center">
  <img src="./TrustCheckAI-demo.gif" width="650">
</p>

TrustCheckAI is an end-to-end bias and compliance auditing, explainability, and model-monitoring platform designed to evaluate bias, mitigate discrimination, explain model decisions, and continuously monitor deployed machine learning systems using Prometheus & Grafana.

It provides real-time dashboards, fairness metrics, model explainability (LIME), drift detection, automated PDF reporting, and user feedback collection — all wrapped in a modern Streamlit UI and containerized for seamless deployment.

---

## 📑 Table of Contents
- ✨ Features
- 🧰 Technical Stack
- 📚 Supported Datasets
- 🏗 System Architecture
- 🔐 Compliance, Fairness & Security
- 🎨 Human-Centered Design (HCI)
- ⚙️ Installation & Setup
- 🧪 Usage Workflow
- 📊 Prometheus Metrics & Grafana Dashboards
- 📉 Drift Detection
- 📘 PDF Report Generation
- 🎥 Demonstration
- 🛣 Roadmap
- 📄 Citations
- 🤝 Acknowledgements

---

## ✨ Features

### 🟣 Bias Detection (AIF360)
- Statistical Parity Difference  
- Disparate Impact  
- Equal Opportunity Difference  
- Equalized Odds  
- Demographic Parity Ratio   

### 🧠 Explainability (XAI)
- **LIME** – local explanations per prediction  

### 📉 Drift Detection
- Kolmogorov–Smirnov (KS) Test  

### ⚙ Model Training & Evaluation
- Logistic Regression  
- Random Forest  
- 5-fold cross-validation  

### 🔎 Real-Time Monitoring & Alerting
- Prometheus metric exporter  
- Grafana dashboards  
- Automated Slack alerts for accuracy/fairness drift  

### 📄 Automatic PDF Reporting
- Full bias report  
- Model performance summary  

### 🖥 Modern Streamlit UI
- Clean, intuitive layout  
- File upload, analysis, visualization  
- User feedback

### 🐳 Fully Containerized
- Streamlit  
- Prometheus  
- Grafana  
- Docker Compose orchestration  

---

## 🧰 Technical Stack

### ML & Fairness
- Python 3.9+  
- Scikit-learn  
- AIF360  
- LIME  

### Monitoring & Observability
- Prometheus  
- Grafana  

### Frontend
- Streamlit  

### DevOps
- Docker  
- Docker Compose  
- GitHub  

---

## 📚 Supported Datasets

### COMPAS – Criminal Justice  
### User Uploaded Structured Dataset

---

## 🏗 System Architecture

```
User Upload → Preprocessing → AIF360 Bias Detection 
        → Model Training → LIME → Drift Detection
        → Prometheus Exporter → Grafana Dashboards & Alerts
```

---

## 🔐 Compliance, Fairness & Security
- Regulatory alignment (EEOC, Justice fairness)  
- Differential privacy  
- Ethical AI lifecycle tracking  
- Secure isolated containers  

---

## 🎨 HCI Principles
- Accessible charts  
- Colorblind-safe design  
- Clear fairness/performance separation  
- Prototyped user flows  

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/27HarshalPatel/TrustCheckAI.git
cd TrustCheckAI
docker-compose up --build
```

Access:
- Streamlit → http://localhost:8501  
- Prometheus → http://localhost:9090  
- Grafana → http://localhost:3000  

---

## 🧪 Usage Workflow
1. Upload CSV  
2. Run fairness analysis  
3. Train models  
4. View LIME 
5. Generate PDF  
6. Monitor in Grafana  
7. Receive alerts if Accuracy falls below 70%

---

## 📊 Prometheus Metrics
- upload_counter  
- analysis_counter  
- accuracy_gauge  
- feedback_ratings_counter  
- feedback_comments_counter  

---

## 📉 Drift Detection
- KS Test 

---

## 📘 PDF Report Generation
Includes fairness metrics and performance summary.

---

## 🎥 Demonstration

<p align="center">
  <img src="./TrustCheckAI-demo.gif" width="700">
</p>

---

## 🛣 Roadmap
- Fairlearn integration  
- Kubernetes deployment  
- Extended fairness metrics  

---

## 📄 Citations
- IBM AIF360  
- COMPAS Dataset  

---

## 🤝 Acknowledgements
- University of Florida  
- HiPerGator Computing  
- Open-source community  
