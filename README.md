# HEART ATTACK ANALYSIS
**Subject:** Data Science Using Python  
**Project Category:** Healthcare Predictive Analytics & Machine Learning Classification  
**Target Audience:** Academic Viva Voce, Examination Submissions, Laboratory Demonstrations

---

## 📌 1. Project Overview & Objectives
Cardiovascular diseases (CVDs) are the leading cause of mortality globally. Early detection of clinical risk indicators allows timely therapeutic and lifestyle intervention.

This project delivers a complete, syllabus-aligned Data Science pipeline in Python:
- **Data Ingestion & Integrity:** Loads and audits the benchmark 14-parameter UCI Cleveland Heart Disease dataset.
- **Exploratory Data Analysis (EDA):** Statistical distributions, class balance, risk correlation matrices, and clinical scatter plots.
- **Data Preprocessing:** Outlier handling, class-stratified train/test splitting, and feature scaling via `StandardScaler` (avoiding data leakage).
- **Machine Learning Benchmark:** Comparative evaluation of four fundamental classification algorithms:
  1. **Logistic Regression** (Interpretable baseline linear classifier)
  2. **Decision Tree** (Rule-based, non-linear classifier)
  3. **Random Forest** (Ensemble bagging classifier)
  4. **K-Nearest Neighbors (KNN)** (Distance-based instance classifier)
- **Clinical Inference Engine:** Live prediction tool calculating probability scores and diagnostic risk recommendations.

---

## 🗂️ 2. Project Directory Structure

```text
heart_disease_predictor/
│
├── data/
│   └── heart_disease.csv          # 303 patient records with 14 clinical features
│
├── output_plots/                  # High-resolution (300 DPI) analysis charts
│   ├── 01_target_distribution.png
│   ├── 02_age_vs_disease.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_chol_vs_bp.png
│   ├── 05_heart_rate_vs_age.png
│   ├── 06_model_accuracy_comparison.png
│   ├── 07_confusion_matrices.png
│   └── 08_feature_importance.png
│
├── Heart_Disease_Analysis.ipynb   # Interactive Jupyter Notebook (pre-executed)
├── main.py                        # Full automated Data Science pipeline
├── predict.py                     # Interactive patient tester for viva/demos
├── requirements.txt               # Dependencies list
└── README.md                      # Documentation & Viva Voce Q&A Cheat Sheet
```

---

## 🩺 3. Dataset Feature Dictionary

| Feature | Type | Description | Clinical Context |
| :--- | :--- | :--- | :--- |
| `age` | Integer | Patient age in years | Risk increases with advancing age |
| `sex` | Binary | 1 = Male, 0 = Female | Biological risk factor |
| `cp` | Categorical | Chest pain type (1: Typical, 2: Atypical, 3: Non-anginal, 4: Asymptomatic) | Primary symptom of myocardial ischemia |
| `trestbps` | Integer | Resting blood pressure (mm Hg on admission) | Hypertension threshold is ≥ 130-140 mm Hg |
| `chol` | Integer | Serum cholesterol in mg/dL | Desirable level is < 200 mg/dL |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dL (1 = True, 0 = False) | Diabetes indicator |
| `restecg` | Categorical | Resting ECG results (0: Normal, 1: ST-T abnormality, 2: LV hypertrophy) | Electrical conductivity of the heart |
| `thalach` | Integer | Maximum heart rate achieved during stress test | Reduced max HR indicates lower cardiovascular reserve |
| `exang` | Binary | Exercise induced angina (1 = Yes, 0 = No) | Chest tightness triggered by exertion |
| `oldpeak` | Float | ST depression induced by exercise relative to rest | Key ECG indicator of myocardial ischemia |
| `slope` | Categorical | Slope of peak exercise ST segment (1: Upsloping, 2: Flat, 3: Downsloping) | Downsloping indicates severe coronary compromise |
| `ca` | Integer | Number of major vessels (0-3) colored by fluoroscopy | Anatomical blockage of coronary arteries |
| `thal` | Categorical | Thalassemia scan (3: Normal, 6: Fixed defect, 7: Reversible defect) | Blood flow / perfusion defect |
| `target` | Binary | **0 = Healthy / Negative, 1 = Heart Disease Detected** | Ground truth diagnosis |

---

## ⚙️ 4. How to Run the Project

### A. Run the Full Automated Pipeline
Executes data loading, statistical summaries, chart generation, model training, and performance benchmarking:
```bash
cd /Users/tanushyadav/.gemini/antigravity-ide/scratch/heart_disease_predictor
python3 main.py
```

### B. Run the Interactive Clinical Predictor (Viva Demo)
Use this to demonstrate live patient classification to an examiner:
```bash
# Test a preset healthy profile:
python3 predict.py healthy

# Test a preset high-risk cardiac profile:
python3 predict.py high

# Or launch the interactive prompt:
python3 predict.py
```

### C. Open the Jupyter Notebook
Open `Heart_Disease_Analysis.ipynb` directly in VS Code, Antigravity IDE, or JupyterLab. All cells are already executed and output visualizations are pre-rendered.

---

## 📊 5. Machine Learning Benchmark Results

On an 80/20 stratified test split ($N_{test} = 61$), models achieved the following benchmark scores:

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **88.52%** | **83.87%** | **92.86%** | **88.14%** | **95.89%** |
| **K-Nearest Neighbors** | 88.52% | 80.00% | 100.00% | 88.89% | 92.32% |
| **Logistic Regression** | 86.89% | 81.25% | 92.86% | 86.67% | 95.13% |
| **Decision Tree** | 78.69% | 74.19% | 82.14% | 77.97% | 85.98% |

> **Top Clinical Influencers (Feature Importance):**
> 1. `thal` (Thalassemia / cardiac perfusion scan)
> 2. `ca` (Number of major blocked coronary vessels)
> 3. `oldpeak` (Exercise-induced ST depression on ECG)
> 4. `thalach` (Maximum heart rate achieved)

---

## 🎓 6. Academic Viva Voce Q&A Cheat Sheet

### Q1: What is the main objective of this project?
**Answer:** The objective is to build a supervised machine learning classification pipeline in Python that analyzes clinical indicators (such as blood pressure, cholesterol, max heart rate, and ECG features) to predict the likelihood of heart disease in patients with high recall and accuracy.

### Q2: Why is this a classification problem rather than a regression problem?
**Answer:** Because the target variable (`target`) is categorical and discrete ($0 = \text{Healthy}$, $1 = \text{Heart Disease}$). If we were predicting a continuous numerical value like patient systolic blood pressure or cholesterol levels, it would be a regression problem.

### Q3: Why is Recall more critical than Precision in medical diagnosis?
**Answer:** In medical screening:
- **False Positive (Type I error):** A healthy patient is diagnosed as sick. They undergo secondary confirmatory testing, causing mild anxiety but no physical harm.
- **False Negative (Type II error):** A sick patient is classified as healthy and sent home without treatment, which can lead to fatal heart attacks.
Recall measures the proportion of actual sick patients correctly identified: $\text{Recall} = \frac{TP}{TP + FN}$. Therefore, maximizing Recall is paramount.

### Q4: Why is `StandardScaler` used and why is it fitted only on `X_train`?
**Answer:** 
1. `StandardScaler` standardizes features to have a mean of 0 and unit variance ($\sigma = 1$). Without scaling, features with large numeric scales (e.g. Cholesterol ~ 250) would disproportionately dominate gradient descent updates in Logistic Regression and distance metrics in KNN over small features like ST depression (0 to 3).
2. We fit the scaler strictly on `X_train` and use `.transform()` on `X_test` to prevent **Data Leakage** (the test set must remain completely unseen during model fitting).

### Q5: What does `stratify=y` do in `train_test_split`?
**Answer:** It ensures that both the training set and testing set have the exact same class distribution ratio (e.g. 54% healthy, 46% disease) as the original dataset, preventing sampling bias.

### Q6: What is the difference between a Decision Tree and a Random Forest?
**Answer:** A Decision Tree is a single rule-based flowchart that easily overfits to training data (high variance). Random Forest is an ensemble of multiple decorrelated decision trees built using **Bootstrap Aggregation (Bagging)** and random feature sub-selection. It aggregates votes across all trees, significantly reducing variance and boosting accuracy.

### Q7: What is a Confusion Matrix?
**Answer:** A $2 \times 2$ matrix comparing actual labels versus predicted labels:
- **True Negatives (TN):** Correctly predicted healthy.
- **False Positives (FP):** Healthy patient misclassified as sick (Type I Error).
- **False Negatives (FN):** Sick patient misclassified as healthy (Type II Error).
- **True Positives (TP):** Correctly predicted heart disease.

### Q8: What does ROC-AUC represent?
**Answer:** The **Receiver Operating Characteristic (ROC)** curve plots the True Positive Rate (Recall) against the False Positive Rate across all classification thresholds. The **Area Under the Curve (AUC)** measures discrimination capability (1.0 = perfect model, 0.5 = random guessing). Our Random Forest model achieved **95.89% ROC-AUC**.

### Q9: What is `oldpeak` and why is it such a strong predictor?
**Answer:** `oldpeak` represents ST depression observed on an electrocardiogram (ECG) during an exercise stress test compared to resting state. Significant ST depression (> 1.0 - 2.0 mm) is a classic physiological sign of myocardial ischemia (lack of oxygenated blood reaching heart muscle under exertion).

### Q10: What is the K-Nearest Neighbors (KNN) algorithm and how does it classify?
**Answer:** KNN is a non-parametric, distance-based algorithm. To classify a new patient, it calculates the Euclidean distance between the patient's scaled feature vector and all training points, identifies the $k$ nearest neighbors (here, $k=5$), and assigns the class through majority voting.

### Q11: How could this project be extended or deployed in real life?
**Answer:**
1. Packaging the model into a REST API using FastAPI or Flask.
2. Integrating into hospital EHR (Electronic Health Record) systems.
3. Adding SHAP (SHapley Additive exPlanations) or LIME to provide patient-specific visual explanations for attending physicians.

---
**Project Prepared for Syllabus Examination & Laboratory Practical Assessment**
