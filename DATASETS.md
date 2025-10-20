# TrustCheckAI - Dataset Documentation

## 📚 Dataset Overview

This project uses two well-established fairness benchmark datasets to evaluate bias and fairness in AI models.

---

## 1. COMPAS Dataset (Criminal Justice)

### Overview
- **Full Name**: Correctional Offender Management Profiling for Alternative Sanctions
- **Domain**: Criminal Justice / Recidivism Prediction
- **Source**: ProPublica Investigation (2016)
- **URL**: https://github.com/propublica/compas-analysis

### Description
The COMPAS dataset contains criminal justice records used to predict recidivism (likelihood of re-offending). This dataset became famous after ProPublica's investigation revealed racial bias in the COMPAS algorithm used by courts in Florida and elsewhere.

### Key Features
- **Sample Size**: ~7,000 records (after filtering)
- **Protected Attribute**: Race (African-American vs Caucasian/Other)
- **Target Variable**: `two_year_recid` (1 = recidivism, 0 = no recidivism)
- **Time Period**: 2013-2014

### Important Columns
| Column | Description | Type |
|--------|-------------|------|
| `age` | Age of defendant | Numeric |
| `sex` | Gender (Male/Female) | Categorical |
| `race` | Race (African-American, Caucasian, etc.) | Categorical |
| `priors_count` | Number of prior crimes | Numeric |
| `c_charge_degree` | Charge degree (F=Felony, M=Misdemeanor) | Categorical |
| `decile_score` | COMPAS risk score (1-10) | Numeric |
| `two_year_recid` | Recidivism within 2 years | Binary |
| `is_recid` | General recidivism flag | Binary |

### Known Bias Issues
- **Higher false positive rates** for African-American defendants
- **Lower false positive rates** for Caucasian defendants
- **Disparate Impact**: African-Americans labeled "high risk" at higher rates

### Citation
```
Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). 
Machine Bias. ProPublica.
https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
```

---

## 2. UCI Adult Income Dataset

### Overview
- **Full Name**: Adult Census Income Dataset
- **Domain**: Finance / Employment / Income Prediction
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/adult

### Description
The Adult dataset contains census data extracted from the 1994 Census database. It's commonly used to predict whether an individual's income exceeds $50K per year based on demographic and employment features.

### Key Features
- **Sample Size**: ~48,000 records
- **Protected Attributes**: 
  - Gender (`sex`: Male/Female)
  - Race (`race`: White, Black, Asian-Pac-Islander, etc.)
- **Target Variable**: `income` (>50K or <=50K)
- **Time Period**: 1994 U.S. Census

### Important Columns
| Column | Description | Type |
|--------|-------------|------|
| `age` | Age in years | Numeric |
| `workclass` | Employment type (Private, Govt, etc.) | Categorical |
| `education` | Education level | Categorical |
| `education-num` | Years of education | Numeric |
| `marital-status` | Marital status | Categorical |
| `occupation` | Job type | Categorical |
| `race` | Race/ethnicity | Categorical |
| `sex` | Gender | Categorical |
| `capital-gain` | Capital gains | Numeric |
| `capital-loss` | Capital losses | Numeric |
| `hours-per-week` | Work hours per week | Numeric |
| `native-country` | Country of origin | Categorical |
| `income` | Income level (>50K or <=50K) | Binary |

### Known Bias Issues
- **Gender disparity**: Males significantly more likely to earn >50K
- **Race disparities**: Income prediction varies across racial groups
- **Intersectional bias**: Compounded effects for minority women

### Citation
```
Kohavi, R. (1996). 
Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid. 
Proceedings of the Second International Conference on Knowledge Discovery and Data Mining.
```

---

## 🔍 Dataset Preprocessing in TrustCheckAI

### COMPAS Preprocessing
1. **Filtering**:
   - Remove records with screening >30 days from arrest
   - Remove ordinary traffic offenses
   - Remove records with missing risk scores

2. **Feature Engineering**:
   - Create binary race variable (African-American = 1, Others = 0)
   - Encode categorical variables (charge degree, gender)
   - Normalize age and prior counts

3. **Train/Test Split**: 70/30 stratified by target variable

### Adult Preprocessing
1. **Cleaning**:
   - Remove records with missing values (marked as '?')
   - Handle whitespace in categorical values

2. **Feature Engineering**:
   - Create binary gender variable (Male = 1, Female = 0)
   - Create binary race variable (White = 1, Others = 0)
   - Encode all categorical features
   - Scale numeric features

3. **Train/Test Split**: 70/30 stratified by target variable

---

## 📊 Data Quality Metrics

### COMPAS Dataset
- **Completeness**: ~95% after filtering
- **Balance**: ~45% recidivism rate
- **Race Distribution**: 
  - African-American: ~51%
  - Caucasian: ~34%
  - Other: ~15%
- **Gender Distribution**: ~80% Male, ~20% Female

### Adult Dataset
- **Completeness**: ~93% after removing missing values
- **Balance**: ~24% earn >50K
- **Gender Distribution**: ~67% Male, ~33% Female
- **Race Distribution**:
  - White: ~85%
  - Black: ~9%
  - Other: ~6%

---

## 🛡️ Ethical Considerations

### COMPAS Dataset
- **Sensitivity**: Contains criminal justice data
- **Privacy**: De-identified but still sensitive
- **Use Case**: Educational/research only
- **Limitations**: 
  - Reflects historical bias in criminal justice system
  - Limited to specific jurisdiction (Broward County, FL)
  - May not generalize to other contexts

### Adult Dataset
- **Age**: Dataset from 1994 - may not reflect current demographics
- **Representation**: Underrepresents certain minority groups
- **Use Case**: Educational/research benchmark
- **Limitations**:
  - Binary gender classification only
  - U.S.-centric (limited international applicability)
  - Income threshold ($50K) outdated

---

## 📥 Data Access in TrustCheckAI

### Automatic Download
Both datasets are **automatically downloaded** when you run the notebook:

```python
# COMPAS - Downloaded from GitHub
compas_url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"

# Adult - Downloaded from UCI Repository
adult_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
```

**No manual download required!**

### Optional: Google Drive Storage
If you want to store datasets on Google Drive:

1. Upload datasets to your Google Drive
2. Get shareable links
3. Use `gdown` to download in notebook:

```python
import gdown
gdown.download('https://drive.google.com/uc?id=YOUR_FILE_ID', 'compas.csv', quiet=False)
```

---

## 📖 Additional Resources

### COMPAS Dataset
- ProPublica Investigation: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
- GitHub Repository: https://github.com/propublica/compas-analysis
- Academic Paper: https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm

### Adult Dataset
- UCI Repository: https://archive.ics.uci.edu/ml/datasets/adult
- Original Paper: http://robotics.stanford.edu/~ronnyk/nbtree.pdf
- Kaggle Version: https://www.kaggle.com/datasets/uciml/adult-census-income

### Fairness Research
- AI Fairness 360: https://aif360.mybluemix.net/
- Fairness Definitions: https://fairware.cs.umass.edu/papers/Verma.pdf
- NIST AI RMF: https://www.nist.gov/itl/ai-risk-management-framework

---

## ⚠️ Important Notes

1. **Research Use Only**: These datasets are for educational/research purposes
2. **No Real-World Deployment**: Do not use these models in production without extensive validation
3. **Historical Bias**: Both datasets contain historical biases that must be acknowledged
4. **Context Matters**: Results may not generalize to other contexts or time periods
5. **Intersectionality**: Multiple protected attributes can compound bias effects

---