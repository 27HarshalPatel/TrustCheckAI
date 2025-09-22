# TrustCheckAI: Bias and Compliance Monitoring for AI Models

## Overview

TrustCheckAI is a prototype system designed to detect, evaluate, and mitigate bias in AI/ML models built on structured datasets. The project focuses on ensuring fairness, transparency, and compliance across domains such as criminal justice, finance, and healthcare.

The system leverages the IBM AI Fairness 360 (AIF360) toolkit for bias detection and fairness metrics, while LIME and SHAP provide interpretability and auditing support.

## Features

Bias & Fairness Evaluation: Measure disparate impact, statistical parity, and equalized odds.

Explainability: Integrate LIME and SHAP to interpret model decisions.

Compliance Monitoring: Generate reports for ethical and regulatory checks.

Dataset Coverage: Test and validate fairness on COMPAS (criminal justice), UCI Adult Income (finance/employment), and optionally MIMIC-III (healthcare).

## Technical Stack

Languages: Python

Frameworks/Tools:

  -> IBM AI Fairness 360

  -> LIME, SHAP

  -> pandas, NumPy, scikit-learn

  -> matplotlib, seaborn

Development Environments:

Local: macOS (Python/Jupyter)

HPC: HiPerGator (for large-scale analysis)
