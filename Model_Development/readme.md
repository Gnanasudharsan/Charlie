# Model Development – README.md

Model Development Module

This directory contains the full end-to-end Machine Learning development workflow for the Charlie MBTA Delay Prediction System.
It includes model training, tuning, selection, sensitivity analysis, bias detection, drift monitoring, experiment tracking, and pushing the final model to Google Cloud Artifact Registry.

⸻

### 1. Folder Structure
```bash
Model_Development/
│
├── src/                     # All ML training, tuning & analysis code
│   ├── model_train.py       # Training scripts for all ML models
│   ├── model_tuning.py      # Hyperparameter tuning (Grid/Random Search)
│   ├── model_select.py      # Model comparison & selection logic
│   ├── bias_analysis.py     # Fairness, slicing, disparate impact
│   ├── model_explain.py     # SHAP & LIME explainability
│   ├── explainability.py    # Feature importance & sensitivity analysis
│   ├── monitor_drift.py     # Drift detection using reference stats
│   ├── register_model.py    # Push model to GCP Artifact Registry
│   ├── gcp_registry.py      # Authentication + upload helper
│   └── utils/               # Shared utilities (logging, helpers)
│
├── models/
│   ├── final_model.joblib       # Deployed model
│   ├── best_logreg_tuned.joblib # Tuned Logistic Regression
│   ├── model_lgbm.joblib        # LightGBM candidate
│
├── reports/
│   ├── model_comparison.json
│   ├── model_comparison.png
│   ├── shap_importance.csv
│   ├── shap_summary.png
│   ├── lime_explanation.html
│   ├── bias_report.csv
│   ├── bias_plot.html
│   ├── drift_report.json
│   └── drift_report.html
│
└── screenshots/              # Visuals for submission
```
⸻

### 2. Overview

This module trains a machine-learning model to predict whether an MBTA bus trip will be delayed based on MBTA real-time API features (direction_id, stop_sequence, etc.).

The pipeline follows production-grade ML engineering practices:
	•	Reproducible training
	•	MLflow experiment tracking
	•	Hyperparameter tuning
	•	Model comparison
	•	Fairness & bias detection
	•	Explainability (SHAP/LIME)
	•	Drift monitoring
	•	GCP Artifact Registry deployment

⸻

### 3. Data Loading

Data used for training comes from the Data_Pipeline module and is version-controlled with DVC.

The loader automatically fetches:

Data_Pipeline/data/processed/latest.parquet

Code:
src/data_loader.py

⸻

### 4. Model Training

Training is executed using:
	•	Logistic Regression (baseline)
	•	Random Forest
	•	LightGBM (final best model)

Key file:
src/model_train.py

Each model logs:
	•	metrics (accuracy, f1, AUC)
	•	confusion matrices
	•	model artifacts
	•	predictions

Tracking: MLflow

⸻

### 5. Hyperparameter Tuning

Performed using GridSearchCV and RandomizedSearchCV.

Code:
src/model_tuning.py

Outputs:
	•	best params
	•	best estimator
	•	tuning logs
	•	comparison plots

All tuning results are stored inside reports/model_comparison.*.

⸻

### 6. Model Selection

After training & tuning, models are compared on:
	•	Accuracy
	•	F1-score
	•	ROC-AUC
	•	Inference speed
	•	Robustness
	•	Drift impact

Code: src/model_select.py

The final chosen model is saved as:

Model_Development/models/final_model.joblib


⸻

### 7. Model Validation

Validation includes:
	•	Hold-out validation
	•	Cross-validation (k=5)
	•	AUC-ROC curves
	•	Confusion matrix
	•	Precision/Recall trade-offs

Metrics are logged via MLflow.

⸻

### 8. Bias Analysis (Fairness)

We perform fairness checks across slices such as:
	•	Direction ID (0 → inbound, 1 → outbound)
	•	Stop sequence ranges
	•	Route type grouping
	•	Time-based segments (AM/PM)

Tool used: Fairlearn Metrics

Code:
src/bias_analysis.py

Artifacts:
	•	bias_report.csv
	•	fairness plots
	•	disparity metrics

⸻

### 9. Explainability (SHAP + LIME)

To understand feature importance:
	•	SHAP (global + local explanations)
	•	LIME (sample-level decision interpretation)

Code:
src/explainability.py

Outputs:
	•	shap_summary.png
	•	shap_importance.csv
	•	lime_explanation.html

⸻

### 10. Drift Detection

Uses historical baseline stats from:

models/reference_stats.json

Drift check steps:
	•	Feature distribution drift
	•	Population stability index
	•	Prediction drift
	•	Anomaly detection thresholds

Code:
src/monitor_drift.py

Reports stored in:
reports/drift_report.*

⸻
### 11. Model Registry – Pushing to GCP

After selection, the model is uploaded to Google Cloud Artifact Registry.

Code:
src/register_model.py

Steps automated:
	1.	Authenticate with service account
	2.	Tag model with version ID
	3.	Upload to:

us-central1-docker.pkg.dev/<PROJECT-ID>/ml-models/



⸻

### 12. CI/CD for Model Development

This module integrates with GitHub Actions + Cloud Build:

✔ Train model on every push
✔ Validate performance thresholds
✔ Run bias checks
✔ Run drift checks
✔ Push new model if performance improves
✔ Rollback if degraded

Pipeline file:
.github/workflows/mlops_pipeline.yml

⸻

### 13. How to Run Locally

1. Activate environment

pip install -r requirements.txt

2. Train all models

python Model_Development/src/model_train.py

3. Run hyperparameter tuning

python Model_Development/src/model_tuning.py

4. Compare models

python Model_Development/src/model_select.py

5. Run bias check

python Model_Development/src/bias_analysis.py

6. Run explainability

python Model_Development/src/explainability.py

7. Push final model to GCP

python Model_Development/src/register_model.py


⸻

