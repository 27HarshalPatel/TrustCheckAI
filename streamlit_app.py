import streamlit as st
import pandas as pd
from prometheus_client import start_http_server, Counter, Gauge #import for Prometheus metrics
import time
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

# AIF360 for Fairness
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing
    AIF360_AVAILABLE = True
except Exception:
    # aif360 can be heavy to install in some environments; gracefully degrade.
    AIF360_AVAILABLE = False

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import lime
import lime.lime_tabular
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def make_pdf_report(initial_report_text: str,
                    performance_df,  # pandas DataFrame from classification_report
                    title: str = "TrustCheckAI – Consolidated Report") -> bytes:
    """
    Returns PDF bytes that contain:
      - Initial bias metrics (SPD, DI)
      - Full sklearn classification report table
      - Optional post-model fairness metrics (EOD/AOD)
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 6))

    # Initial Bias section
    story.append(Paragraph("Initial Bias Report", styles["Heading2"]))
    # Preserve line breaks from your text report
    for line in (initial_report_text or "").splitlines():
        story.append(Paragraph(line or "&nbsp;", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Model Performance section
    story.append(Paragraph("Model Performance Report", styles["Heading2"]))

    # Ensure consistent column order if present
    cols = [c for c in ["precision", "recall", "f1-score", "support"] if c in performance_df.columns]
    rows = [["label"] + cols]
    for idx, row in performance_df[cols].iterrows():
        # idx can be 'accuracy' row (with precision only). Convert to strings safely.
        vals = []
        for c in cols:
            val = row[c]
            try:
                vals.append(f"{float(val):.4f}")
            except Exception:
                vals.append(str(val))
        rows.append([str(idx)] + vals)

    table = Table(rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# Prometheus metrics
@st.cache_resource
def get_prometheus_metrics():
    start_http_server(8000)
    upload_counter = Counter('file_uploads_total', 'Total number of uploaded files')
    analysis_counter = Counter('analysis_requests_total', 'Total number of analysis requests')
    accuracy_gauge = Gauge('model_accuracy', 'Model accuracy for the last run analysis')
    feedback_ratings_counter = Counter('feedback_ratings_total', 'Total number of feedback ratings', ['rating'])
    feedback_comments_counter = Counter('feedback_comments_total', 'Total number of feedback comments')
    return upload_counter, analysis_counter, accuracy_gauge, feedback_ratings_counter, feedback_comments_counter

UPLOAD_COUNTER, ANALYSIS_COUNTER, ACCURACY_GAUGE, FEEDBACK_RATINGS_COUNTER, FEEDBACK_COMMENTS_COUNTER = get_prometheus_metrics()

# --- Helper functions ---

def download_and_load_compas():
    compas_url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    urllib.request.urlretrieve(compas_url, 'compas-scores-two-years.csv')
    compas_df = pd.read_csv('compas-scores-two-years.csv', low_memory=False)

    compas_clean = compas_df[
        (compas_df['days_b_screening_arrest'] <= 30) &
        (compas_df['days_b_screening_arrest'] >= -30) &
        (compas_df['is_recid'] != -1) &
        (compas_df['c_charge_degree'] != 'O') &
        (compas_df['score_text'] != 'N/A')
    ].copy()

    compas_features = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text',
                       'sex', 'priors_count', 'days_b_screening_arrest',
                       'decile_score', 'is_recid', 'two_year_recid']
    compas_clean = compas_clean[compas_features].dropna().copy()

    compas_clean['target'] = compas_clean['two_year_recid']
    compas_clean['race_binary'] = (compas_clean['race'] == 'African-American').astype(int)

    for col in compas_clean.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        compas_clean[col + '_encoded'] = le.fit_transform(compas_clean[col])

    return compas_clean

def preprocess_uploaded_data(df, protected_attribute, target_variable):
    """Generic preprocessing for uploaded datasets."""
    df_clean = df.dropna().copy()
    if df_clean.empty:
        return df_clean

    # Convert protected attribute to binary
    unprivileged_value = df_clean[protected_attribute].value_counts().idxmin()
    df_clean[protected_attribute] = (df_clean[protected_attribute] != unprivileged_value).astype(int)

    # Convert target variable to binary
    favorable_value = df_clean[target_variable].value_counts().idxmax()
    df_clean[target_variable] = (df_clean[target_variable] == favorable_value).astype(int)

    # Encode all object-type columns
    for col in df_clean.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_clean[col + '_encoded'] = le.fit_transform(df_clean[col])

    return df_clean


def initial_bias_analysis(df, protected_attribute, target_variable):
    if not AIF360_AVAILABLE:
        return "AIF360 not installed; initial bias analysis unavailable.", (None, None)
    try:
        df_numeric = df.select_dtypes(include='number')
        dataset = BinaryLabelDataset(
            favorable_label=0, unfavorable_label=1,
            df=df_numeric,
            label_names=[target_variable],
            protected_attribute_names=[protected_attribute]
        )
        metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{protected_attribute: 1}], privileged_groups=[{protected_attribute: 0}])
        spd = metric.statistical_parity_difference()
        di = metric.disparate_impact()
        report = f"Statistical Parity Difference: {spd:.4f}\nDisparate Impact: {di:.4f}"
        return report, (spd, di)
    except (ZeroDivisionError, RuntimeWarning) as e:
        return f"Could not compute initial bias metrics: {e}", (None, None)

def apply_reweighing(df, protected_attribute, target_variable):
    if not AIF360_AVAILABLE:
        return "AIF360 not installed; reweighing unavailable.", (None, None)
    try:
        df_numeric = df.select_dtypes(include='number')
        dataset = BinaryLabelDataset(
            favorable_label=0, unfavorable_label=1,
            df=df_numeric,
            label_names=[target_variable],
            protected_attribute_names=[protected_attribute]
        )
        RW = Reweighing(unprivileged_groups=[{protected_attribute: 1}], privileged_groups=[{protected_attribute: 0}])
        dataset_transformed = RW.fit_transform(dataset)
        metric_transformed = BinaryLabelDatasetMetric(dataset_transformed, unprivileged_groups=[{protected_attribute: 1}], privileged_groups=[{protected_attribute: 0}])
        spd = metric_transformed.statistical_parity_difference()
        di = metric_transformed.disparate_impact()
        report = f"Statistical Parity Difference: {spd:.4f}\nDisparate Impact: {di:.4f}"
        return report, (spd, di)
    except (ZeroDivisionError, RuntimeWarning) as e:
        return f"Could not compute mitigated bias metrics: {e}", (None, None)

def train_and_evaluate_model(df, protected_attribute, target_variable):
    try:
        exclude_cols = {
            target_variable,
            protected_attribute,
            'target',                # prevent target leakage from COMPAS helper
            'two_year_recid',
            'is_recid',
            'decile_score',
            }
        feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols
        ]

        X = df[feature_cols]
        y = df[target_variable]

        if y.value_counts().min() < 2:
            return "Error: The target variable must have at least two samples for each class to perform a stratified split.", None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        best_model_name = ""
        best_model = None
        best_accuracy = 0
        best_report = None
        best_artifacts = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                best_report = pd.DataFrame(report).transpose()
                best_artifacts = {
                    "model": best_model, "scaler": scaler, "X_train_scaled": X_train_scaled,
                    "X_test_scaled": X_test_scaled, "X_test": X_test, "y_test": y_test,
                    "y_pred": y_pred, "feature_names": feature_cols, "model_name": best_model_name,
                }

        return best_report, best_artifacts
    except ValueError as e:
        return f"Error during model training: {e}", None

def post_model_fairness_evaluation(df, protected_attribute, target_variable, artifacts):
    if not AIF360_AVAILABLE:
        return "AIF360 not installed; post-model fairness evaluation unavailable.", (None, None)
    try:
        test_df = artifacts['X_test'].copy()
        test_df[target_variable] = artifacts['y_test']
        test_df[protected_attribute] = df.loc[test_df.index, protected_attribute]

        df_numeric = test_df.select_dtypes(include='number')

        test_dataset = BinaryLabelDataset(
            favorable_label=0, unfavorable_label=1,
            df=df_numeric,
            label_names=[target_variable],
            protected_attribute_names=[protected_attribute]
        )
        pred_dataset = test_dataset.copy(deepcopy=True)
        pred_dataset.labels = artifacts['y_pred'].reshape(-1, 1)

        metric = ClassificationMetric(
            test_dataset, pred_dataset,
            unprivileged_groups=[{protected_attribute: 1}],
            privileged_groups=[{protected_attribute: 0}]
        )
        eod = metric.equal_opportunity_difference()
        aod = metric.average_abs_odds_difference()
        report = f"Equal Opportunity Difference: {eod:.4f}\n"
        report += f"Average Absolute Odds Difference: {aod:.4f}"
        return report, (eod, aod)
    except (ZeroDivisionError, RuntimeWarning) as e:
        return f"Could not compute post-model fairness metrics: {e}", (None, None)

def explain_model_with_lime(artifacts):
    plt.clf()
    model = artifacts['model']
    X_train_scaled = artifacts['X_train_scaled']
    X_test_scaled = artifacts['X_test_scaled']
    feature_names = artifacts['feature_names']

    # LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_scaled,
        feature_names=feature_names,
        class_names=['Not Recidivist', 'Recidivist'],
        discretize_continuous=True
    )

    st.subheader("Explanation for a few instances")

    # Explain a few instances
    for i in range(2):
        exp = explainer.explain_instance(
            X_test_scaled[i],
            model.predict_proba,
            num_features=5
        )

        st.write(f"Instance {i+1}:")
        st.write(f"Model prediction: {'Recidivist' if model.predict(X_test_scaled[i].reshape(1, -1))[0] == 1 else 'Not Recidivist'}")

        fig = exp.as_pyplot_figure()
        st.pyplot(fig, width='stretch')
    
def main():
    st.title("TrustCheckAI - Bias & Compliance Detector")

    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    df = None  # Use a local variable for the loaded dataframe in this run.

    file_upload = st.file_uploader("Upload Dataset (CSV or Excel)", on_change=lambda: st.session_state.update(analysis_results=None))
    if file_upload:
        try:
            file_size_mb = file_upload.size / (1024 * 1024)
            if file_upload.name.endswith('.csv'):
                df = pd.read_csv(file_upload)
            elif file_upload.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_upload)
            else:
                st.error("Unsupported file format.")

            if file_size_mb > 10:
                st.warning("File size is greater than 10 MB. The dataset will be sampled for faster processing.")
                df = df.sample(frac=0.1, random_state=42)

        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    # --- Analysis Section ---
    if df is not None:
        st.header("Data Preview")
        st.dataframe(df.head(5))
        protected_attribute = st.selectbox("Select Protected Attribute", list(df.columns))
        target_variable = st.selectbox("Select Target Variable", list(df.columns))

        if st.button("Analyze Dataset"):
            ANALYSIS_COUNTER.inc()

            df_processed = preprocess_uploaded_data(df, protected_attribute, target_variable)
            if df_processed.empty:
                st.error("The dataset is empty after removing rows with missing values. Please upload a dataset with more data.")
                return
            initial_report, (spd, di) = initial_bias_analysis(df_processed, protected_attribute, target_variable)

            performance_report, artifacts = train_and_evaluate_model(df_processed, protected_attribute, target_variable)

            st.session_state.analysis_results = {
                "initial_report": initial_report,
                "spd": spd,
                "di": di,
                "performance_report": performance_report,
                "artifacts": artifacts
            }

        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            initial_report = results["initial_report"]
            spd = results["spd"]
            di = results["di"]
            performance_report = results["performance_report"]
            artifacts = results["artifacts"]

            st.header("Bias Check")
            if spd is not None and di is not None:
                if -0.1 < spd < 0.1 and 0.8 < di < 1.2:
                    st.success("Pass, The Dataset is Fair")
                else:
                    st.error("Fail, The Dataset is Biased")
            else:
                st.error("Fail, The Dataset is Biased")

            st.header("Compliance Check")
            if spd is not None and di is not None:
                if -0.1 < spd < 0.1 and 0.8 < di < 1.2:
                    st.success("Pass")
                else:
                    st.error("Fail")
            else:
                st.error("Fail")
            
            # Build the PDF
            pdf_bytes = make_pdf_report(
                initial_report_text=initial_report,
                performance_df=performance_report,
            )

            st.subheader("Download Bias And Performance Report")
            st.download_button(
                data=pdf_bytes,
                label="⬇️ Download Report.pdf",
                file_name="Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            if artifacts:
                accuracy = performance_report.loc['accuracy', 'precision']
                ACCURACY_GAUGE.set(accuracy)
                acc = accuracy * 100
                st.subheader(f"Accuracy: {acc:.2f} %")

            if artifacts is not None:
                st.header("LIME Analysis")
                st.markdown("""
                **LIME (Local Interpretable Model-agnostic Explanations)** is a technique that explains the predictions of any classifier in an interpretable and faithful manner by learning an interpretable model locally around the prediction.
                """)
                explain_model_with_lime(artifacts)
            else:
                st.error(performance_report)

            st.header("Recommendations")
            if spd is not None and di is not None:
                if not (-0.1 < spd < 0.1 and 0.8 < di < 1.2):
                    st.markdown("""
                    - **Investigate Data:** Further examine the data to understand the source of the bias.
                    - **Consider Different Mitigation Techniques:** Reweighing is just one option. You might also consider other pre-processing, in-processing, or post-processing techniques.
                    - **Feature Engineering:** Carefully review and select features to ensure they are not proxies for the protected attribute.
                    """)
                else:
                    st.markdown("- The dataset appears to be fair. No specific recommendations at this time.")

            st.subheader("Drift Detection")
            st.markdown("""
            We use the Kolmogorov–Smirnov (KS) test to detect data drift between the
            training (reference) data and the current (test) data.

            For each numeric feature, we:
            - Compare the distribution in training vs. test
            - Compute the KS statistic (0–1, higher = more drift)
            - Compute a p-value (p < 0.05 ⇒ statistically significant drift)
            """)

            if artifacts:
                X_train = pd.DataFrame(artifacts['X_train_scaled'], columns=artifacts['feature_names'])
                X_test = pd.DataFrame(artifacts['X_test_scaled'], columns=artifacts['feature_names'])

                alpha = 0.05  # significance level
                drift_rows = []

                for col in artifacts['feature_names']:
                    train_vals = X_train[col].values
                    test_vals = X_test[col].values

                    stat, p_value = ks_2samp(train_vals, test_vals)
                    drift_detected = p_value < alpha

                    drift_rows.append({
                        "feature": col,
                        "ks_statistic": stat,
                        "p_value": p_value,
                        "drift_detected": drift_detected,
                    })

                drift_df = pd.DataFrame(drift_rows)
                st.dataframe(drift_df)
            else:
                st.info("Please train a model and upload data to run drift detection.")

    else:
        st.info("Please upload a dataset to begin analysis.")
    
    # --- Feedback Section ---
    st.subheader("Feedback")
    

    rating = st.slider("Rate your experience (1-5)", 1, 5, 3)
    feedback = st.text_area("Provide your feedback here:")
    if st.button("Submit Feedback"):
        FEEDBACK_RATINGS_COUNTER.labels(rating=str(rating)).inc()
        if feedback:
            FEEDBACK_COMMENTS_COUNTER.inc()
        with open("feedback.log", "a") as f:
            f.write(f"Rating: {rating}\nFeedback: {feedback}\n")
        st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
