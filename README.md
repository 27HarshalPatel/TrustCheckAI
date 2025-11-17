# TrustCheckAI - Bias & Compliance Detector (Streamlit Edition)

A comprehensive web-based platform for AI fairness evaluation and bias detection, rebuilt with a modern Streamlit and Docker-based architecture for simplified deployment and monitoring. This tool allows users to upload their datasets, analyze them for bias, and receive recommendations for mitigation.

## 🚀 Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/TrustCheckAI.git
    cd TrustCheckAI
    ```

2.  **Launch the application with Docker Compose:**
    ```bash
    docker-compose up --build
    ```

3.  **Access the application and monitoring tools:**
    *   **Streamlit App:** `http://localhost:7860`
    *   **Prometheus:** `http://localhost:9090`
    *   **Grafana:** `http://localhost:3000`

## Key Features

-   **📊 Upload Any Dataset**: Supports CSV or Excel files.
-   **🔍 Bias & Compliance Analysis**: In-depth bias and compliance detection using the AIF360 toolkit. The application calculates metrics like Statistical Parity Difference and Disparate Impact.
-   **🤖 Model Training and Evaluation**: Trains Logistic Regression and Random Forest models on the uploaded data and evaluates their performance.
-   **📈 Integrated Monitoring**: Real-time application monitoring with Prometheus and Grafana. Key metrics include file uploads, analysis requests, and model accuracy.
-   **🐳 Dockerized Deployment**: Easy and consistent deployment with Docker Compose. The entire environment, including the Streamlit application, Prometheus, and Grafana, is containerized.
-   **🎨 Modern UI**: Interactive and user-friendly interface with Streamlit.
-   **📝 User Feedback**: Collects user feedback to improve the application.
-   **📄 PDF Report Generation**: Generates a downloadable PDF report with the bias analysis and model performance metrics.
-   **🧠 LIME for Explainability**: Integrates LIME (Local Interpretable Model-agnostic Explanations) to explain the predictions of the trained models.
-   **🌊 Drift Detection**: Uses the Kolmogorov-Smirnov (KS) test to detect data drift between the training and test datasets.

## 🎥 Demonstration

![TrustCheckAI Demo](./TrustCheckAI-Demo.gif)


## Project Structure

```
.
├── Dockerfile
├── README.md
├── streamlit_app.py
├── docker-compose.yml
├── prometheus.yml
└── requirements.txt
```

## Setup Instructions

### Prerequisites

-   Docker
-   Docker Compose

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/TrustCheckAI.git
    cd TrustCheckAI
    ```

2.  **Build and run the containers:**
    ```bash
    docker-compose up --build
    ```

## Usage

1.  **Open the Streamlit application** in your web browser at `http://localhost:7860`.
2.  **Upload a dataset** in CSV or Excel format.
3.  **Select the protected attribute and target variable** from the dropdowns.
4.  **Click the "Analyze Dataset" button** to view the data preview and analysis results.
5.  **Download the PDF report** with the bias analysis and model performance metrics.
6.  **Explore the LIME explanations** for the model's predictions.
7.  **Provide feedback** in the text area and click "Submit Feedback".

## Monitoring

The application is instrumented with Prometheus to collect key metrics. Grafana can be used to visualize these metrics in a dashboard.

### Prometheus

-   **Access the Prometheus UI** at `http://localhost:9090`.
-   **Query for metrics** such as:
    -   `file_uploads_total`: Total number of uploaded files.
    -   `analysis_requests_total`: Total number of analysis requests.
    -   `model_accuracy`: Model accuracy for the last run analysis.
    -   `feedback_ratings_total`: Total number of feedback ratings.
    -   `feedback_comments_total`: Total number of feedback comments.

### Grafana

1.  **Access the Grafana UI** at `http://localhost:3000`.
2.  **Log in** with the default credentials (admin/admin).
3.  **Add a new data source:**
    *   Select "Prometheus".
    *   Set the URL to `http://prometheus:9090`.
    *   Click "Save & Test".
4.  **Create a new dashboard** to visualize the Prometheus metrics. You can create panels for each metric to monitor the application's performance and usage.

## Troubleshooting

-   **Ensure Docker and Docker Compose are installed correctly.**
-   **Verify that the required ports (7860, 8000, 9090, 3000) are not in use by other applications.**
-   **Check the container logs for any errors:**
    ```bash
    docker-compose logs -f
    ```
