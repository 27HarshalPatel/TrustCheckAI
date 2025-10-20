# TrustCheckAI: Bias and Compliance Monitoring for AI Models

## 🎯 Project Overview

**TrustCheckAI** is a comprehensive system for detecting and evaluating bias and fairness in AI/ML models developed with structured datasets. The project focuses on identifying and mitigating unintended bias in input data and model predictions, ensuring fair and equitable AI outcomes.

### Key Objectives
- ✅ Detect bias in structured datasets (COMPAS, UCI Adult Income)
- ✅ Evaluate fairness using IBM AIF360 metrics
- ✅ Provide model explainability using SHAP and LIME
- ✅ Generate compliance reports for regulatory requirements
- ✅ Implement automated bias mitigation strategies

---

## 📊 Datasets

### 1. **COMPAS Dataset** (Criminal Justice)
- **Source**: ProPublica
- **Description**: Recidivism prediction data from the criminal justice system
- **Protected Attribute**: Race (African-American vs Others)
- **Target**: Two-year recidivism prediction
- **Size**: ~7,000 records

### 2. **UCI Adult Income Dataset** (Finance/Employment)
- **Source**: UCI Machine Learning Repository
- **Description**: Census data for income prediction
- **Protected Attributes**: Sex (Male vs Female), Race
- **Target**: Income level (>50K or <=50K)
- **Size**: ~48,000 records

---

## 🛠️ Technical Stack

### Core Frameworks
- **IBM AI Fairness 360 (AIF360)**: Bias detection and mitigation
- **SHAP**: Model explainability for tree-based models
- **LIME**: Local interpretable model-agnostic explanations
- **scikit-learn**: Machine learning models
- **pandas/NumPy**: Data processing

### Fairness Metrics Implemented
| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Statistical Parity Difference** | Measures difference in positive prediction rates between groups | -0.10 to 0.10 |
| **Disparate Impact** | Ratio of positive rates (80% rule) | 0.80 to 1.25 |
| **Equal Opportunity Difference** | Difference in true positive rates | -0.10 to 0.10 |
| **Average Absolute Odds Difference** | Average of TPR and FPR differences | < 0.10 |

### Models Used
- **COMPAS Dataset**: Logistic Regression
- **Adult Dataset**: Random Forest Classifier

---

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- 8GB+ RAM recommended
- 5GB free disk space

### Option 1: Using Docker (Recommended) -> ( Yet to be implemented)


### Option 2: Local Installation

1. **Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

4. **Open and Run**
- Open `TrustCheckAI-playground.ipynb`
- Execute all cells in order

---

## 📁 Project Structure

```
trustcheckai/
│
├── TrustCheckAI-playground.ipynb    # Main notebook with complete pipeline
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/                          # Downloaded datasets (auto-created)
├── outputs/                       # Generated reports and visualizations
├── models/                        # Saved model artifacts
│

---

## 📖 Implementation Pipeline

### Step-by-Step Workflow

#### 1️⃣ **Environment Setup**
- Install all required packages
- Import libraries and configure settings

#### 2️⃣ **Data Collection**
- Automatically download COMPAS dataset from ProPublica
- Automatically download UCI Adult dataset from UCI repository
- Both datasets are loaded directly via URLs (no manual download needed)

#### 3️⃣ **Data Preprocessing & Cleaning**
- Handle missing values
- Filter datasets per standard analysis criteria
- Encode categorical variables
- Define protected attributes
- Create binary target variables

#### 4️⃣ **Exploratory Data Analysis (EDA)**
- Comprehensive visualizations for both datasets
- Distribution analysis of protected attributes
- Target variable analysis
- Correlation studies

#### 5️⃣ **Initial Bias Detection (Pre-Model)**
- Calculate baseline fairness metrics using AIF360
- Assess Statistical Parity Difference
- Evaluate Disparate Impact
- Analyze base rates across groups

## Yet to be implemented

#### 6️⃣ **Bias Mitigation**
- Apply AIF360 Reweighing algorithm
- Transform datasets to reduce bias
- Re-evaluate fairness metrics post-mitigation

#### 7️⃣ **Model Training**
- Train Logistic Regression on COMPAS dataset
- Train Random Forest on Adult dataset
- Evaluate model performance (accuracy, ROC-AUC)
- Generate classification reports

#### 8️⃣ **Post-Model Fairness Evaluation**
- Comprehensive fairness metric calculation
- Statistical Parity Difference assessment
- Disparate Impact evaluation
- Equal Opportunity analysis
- Average Absolute Odds Difference

#### 9️⃣ **Model Explainability**
- **SHAP Analysis**: Feature importance for COMPAS model
- **LIME Analysis**: Instance-level explanations for Adult model
- Visualize feature contributions

#### 🔟 **Comprehensive Reporting**
- Generate fairness comparison visualizations
- Create JSON evaluation report
- Export compliance documentation
- Save all artifacts

#### 1️⃣1️⃣ **Export Artifacts**
- Trained models (.pkl files)
- Scalers and preprocessors
- Processed datasets
- Evaluation reports (JSON + visualizations)

---

## 📊 Expected Outputs

### Generated Files

| File | Description |
|------|-------------|
| `compas_model.pkl` | Trained Logistic Regression model for COMPAS |
| `adult_model.pkl` | Trained Random Forest model for Adult dataset |
| `compas_scaler.pkl` | Feature scaler for COMPAS dataset |
| `adult_scaler.pkl` | Feature scaler for Adult dataset |
| `compas_processed.csv` | Cleaned and preprocessed COMPAS data |
| `adult_processed.csv` | Cleaned and preprocessed Adult data |
| `trustcheckai_evaluation_report.json` | Comprehensive fairness metrics report |
| `fairness_evaluation_report.png` | Visual comparison of fairness metrics |

### Visualizations

1. **EDA Visualizations**
   - Target distribution plots
   - Protected attribute distributions
   - Age/demographic histograms
   - Cross-tabulations (Recidivism by Race, Income by Gender)

2. **Fairness Comparison Charts**
   - Statistical Parity Difference comparison
   - Disparate Impact evaluation
   - Equal Opportunity metrics
   - Model performance comparison

3. **Explainability Outputs**
   - SHAP summary plots (feature importance)
   - LIME explanation charts (instance-level)

---

## 🎓 Key Features

### ✅ Automated Pipeline
- End-to-end automated execution
- No manual intervention required
- Automatic dataset download and preprocessing

### ✅ Comprehensive Bias Detection
- Pre-model bias analysis
- Post-model fairness evaluation
- Multiple fairness metrics
- Threshold-based compliance checking

### ✅ Bias Mitigation
- AIF360 Reweighing algorithm
- Automatic bias reduction
- Before/after comparison

### ✅ Model Explainability
- SHAP for global feature importance
- LIME for local instance explanations
- Visual interpretability

### ✅ Compliance Reporting
- JSON-formatted evaluation reports
- Visual fairness dashboards
- Pass/fail status for each metric
- Regulatory threshold compliance

---

### Adding Additional Datasets

1. Download your dataset
2. Follow the preprocessing pattern in Section 3
3. Define protected attributes
4. Apply the same pipeline (Sections 5-11)

### Changing Models

Replace model definitions in Section 7:

```python
# Example: Switch to XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(random_state=42)
```

---

## 📈 Performance Benchmarks

### Expected Execution Times

| Stage | COMPAS | Adult | Notes |
|-------|--------|-------|-------|
| Data Download | ~10s | ~5s | Network-dependent |
| Preprocessing | ~5s | ~20s | Dataset size dependent |
| Initial Bias Analysis | ~3s | ~8s | AIF360 computation |
{ Yet to be implemented }
| Bias Mitigation | ~5s | ~15s | Reweighing algorithm |
| Model Training | ~10s | ~30s | Random Forest takes longer |
| Fairness Evaluation | ~5s | ~10s | Comprehensive metrics |
| Explainability (SHAP/LIME) | ~20s | ~30s | Most time-intensive |
| **Total** | **~2-3 min** | **~5-8 min** | End-to-end execution |

### Resource Requirements

- **RAM**: 4-8GB recommended
- **CPU**: Multi-core recommended for Random Forest
- **Disk**: ~2GB for datasets and artifacts
- **GPU**: Optional (not required for structured data)

---

## 🛡️ Compliance & Ethics

### Regulatory Alignment

This project implements fairness metrics aligned with:
- **EEOC Guidelines** (Employment equity)
- **COMPAS Fairness Standards** (Criminal justice)
- **HIPAA Considerations** (Healthcare data - optional MIMIC-III)

### Ethical Considerations

- ✅ Transparent bias detection methodology
- ✅ Explainable AI outcomes
- ✅ Protected attribute handling
- ✅ Audit trail generation
- ✅ Stakeholder-friendly reporting

### Privacy & Security

- Datasets used are publicly available
- No personally identifiable information (PII) collected
- All processing done locally
- Optional: Differential privacy for sensitive datasets

---

## 📚 References & Citations

### Datasets
- **COMPAS**: Angwin, J., et al. (2016). "Machine Bias." ProPublica.
- **UCI Adult**: Kohavi, R. (1996). "Scaling Up the Accuracy of Naive-Bayes Classifiers." UCI ML Repository.

### Tools & Frameworks
- **AIF360**: Bellamy et al. (2019). "AI Fairness 360: An Extensible Toolkit for Detecting and Mitigating Algorithmic Bias." IBM Journal of Research and Development.
- **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
- **LIME**: Ribeiro, M. T., et al. (2016). "Why Should I Trust You?" ACM SIGKDD.

## ⚠️ Known Limitations

1. **Scope**: Limited to structured datasets (tabular data)
2. **Datasets**: Currently implements COMPAS and Adult (MIMIC-III optional)
3. **Fairness Metrics**: Focuses on group fairness (not individual fairness)
4. **Scalability**: Designed for datasets <100K records
5. **Deployment**: Prototype stage (not production-ready)

---

## 🔮 Future Enhancements

- [ ] Add MIMIC-III healthcare dataset support
- [ ] Implement real-time bias drift monitoring (NannyML)
- [ ] Add web-based dashboard (Streamlit/Gradio)
- [ ] Support for unstructured data (images, text)
- [ ] API endpoints for model serving
- [ ] Automated retraining pipelines
- [ ] Multi-stakeholder feedback integration

---

## 📞 Support & Contact

For questions or issues:
- 📧 Email: harshal27patel@gmail.com
- 🐛 Issues: GitHub Issues section
- 📖 Documentation: This README + inline notebook comments

---

## 🎓 Academic Context

**Course**: Artificial Intelligence System  
**Semester**: Fall 2025  
**Institution**: University of Florida
**Instructor**: Andrea Ramirez Salgado

---

## 🚀 Getting Started Checklist

- [ ] Clone/download project files
- [ ] Open Jupyter Notebook
- [ ] Open `TrustCheckAI-playground.ipynb`
- [ ] Run all cells sequentially
- [ ] Review generated reports in `outputs/`
- [ ] Check saved models in `models/`

---
