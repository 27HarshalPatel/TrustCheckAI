# 🚀 TrustCheckAI — Bias and Compliance Detaction Platform

<p align="center">
  <img src="./TrustCheckAI-demo.gif" width="650">
</p>

TrustCheckAI is an end-to-end bias and compliance auditing, explainability, and model-monitoring platform designed to evaluate bias, mitigate discrimination, explain model decisions, and continuously monitor deployed machine learning systems using Prometheus & Grafana.

It provides real-time dashboards, fairness metrics, model explainability (LIME), drift detection, automated PDF reporting, and user feedback collection — all wrapped in a modern Streamlit UI and containerized for seamless deployment.

---

## 📑 Table of Contents
- ✨ Features
- 📚 Project Structure
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

## 📚 Project Structure

```text
TrustCheckAI/
├── .ipynb_checkpoints/        # Auto-generated Jupyter checkpoints
├── __pycache__/               # Python bytecode cache
├── .DS_Store                  # macOS system metadata
├── Dockerfile                 # Docker build instructions
├── Final Report.pdf           # Final Project Report Template 
├── README.md                  # Project documentation
├── TrustCheckAI-Demo.mp4      # Full application demo video
├── TrustCheckAI-demo.gif      # GIF preview for README
├── compas-scores-two-years.csv # COMPAS dataset for fairness analysis
├── docker-compose.yml         # Multi-service orchestration (Streamlit + Prometheus + Grafana)
├── feedback.log               # Logs for user feedback & events
├── prometheus.yml             # Prometheus scraping config
├── requirements.txt           # Python dependencies
└── streamlit_app.py           # Main Streamlit application
```
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

Each dataset includes at least one **protected attribute** such as race, gender, or age that is used for fairness auditing.
---

## 🏗 System Architecture

The high-level architecture of TrustCheckAI is shown below:

```txt
                   +---------------------------+
                   |        User (UI)         |
                   |  • Upload CSV dataset    |
                   |  • Configure analysis    |
                   +-------------+------------+
                                 |
                                 v
                     +-----------+-----------+
                     |  Streamlit Application |
                     |  • Orchestration       |
                     |  • UX & controls       |
                     +-----------+------------+
                                 |
         +-----------------------+------------------------+
         |                        |                        |
         v                        v                        v
+----------------+     +-----------------------+   +---------------------+
| Preprocessing  |     | Bias & Fairness       |   | Model Training &    |
| & Validation   |---->| Analysis (AIF360)     |-->| Evaluation (SKL)    |
| • Cleaning     |     | • Metrics & thresholds|   | • LR / RF           |
+----------------+     +-----------------------+   +---------------------+
                                                             |
                                                             v
                                              +-----------------------------+
                                              | Explainability (LIME)       |
                                              +-----------------------------+
                                                             |
                                                             v
                                              +-----------------------------+
                                              | Drift Detection (KS)        |
                                              +-----------------------------+
                                                             |
                                                             v
                                           +-----------------+-----------------+
                                           | Prometheus Metrics Exporter      |
                                           | • upload_counter, accuracy_gauge |
                                           +-----------------+-----------------+
                                                             |
                                                             v
                                     +------------------------+---------------------+
                                     |          Grafana Dashboards & Alerts        |
                                     |  • Accuracy / fairness panels               |
                                     |  • Slack / email alerts                     |
                                     +----------------------------------------------+
```

**Component summary:**  
- **Streamlit App** – central controller for data upload, analysis steps, and visualization.  
- **AIF360 Module** – computes fairness metrics and applies mitigation algorithms.  
- **Model Training** – trains ML models and logs metrics.  
- **XAI Module** – generates LIME explanations for transparency.  
- **Drift Detection** – monitors changes in data and predictions over time.  
- **Prometheus & Grafana** – collect, visualize, and alert on key metrics.

---

## 🧩 Protected Attribute

In TrustCheckAI, the **protected attribute** is a sensitive feature such as **race, gender, age, or ethnicity** that represents groups we want to **protect from unfair treatment**.

Why it is important:

- 📏 **Fairness metrics are defined with respect to protected groups.**  
  Measures like Statistical Parity Difference, Disparate Impact, and Equal Opportunity compare outcomes between protected and non‑protected groups. Without a protected attribute, these metrics cannot be computed.

- 🧪 **Bias detection requires group-wise comparison.**  
  By conditioning on the protected attribute, TrustCheckAI can reveal whether the model treats one group systematically worse than another (e.g., lower approval rates or higher false-positive rates).

- 🛡 **Used for auditing, not for discrimination.**  
  In a responsible workflow, the protected attribute is often **excluded from the model features used for prediction**, but **retained in the evaluation pipeline** so that fairness can be audited post‑hoc.

- 📜 **Regulatory and ethical compliance.**  
  Many regulations (EEOC, GDPR “special categories”, anti‑discrimination laws) explicitly refer to protected characteristics. Correctly identifying and handling the protected attribute is essential for demonstrating compliance.

TrustCheckAI makes the protected attribute explicit in the UI and in the generated reports so that stakeholders clearly understand **which groups are being evaluated for fairness** and how mitigation affects them.

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
2. Select "Protected" Attribute
3. Select "Target" Variable
4. Run "Analyze Dataset" 
5. View the Bias and Compliance Check Result along with Accuracy in Predicting the Results
6. View LIME Analyses
7. Generate PDF  
8. Monitor in Grafana  
9. Receive alerts if Accuracy falls below 70%

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
