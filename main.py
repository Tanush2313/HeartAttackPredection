"""
=============================================================================
 PROJECT TITLE: HEART ATTACK ANALYSIS
 SUBJECT      : Data Science Using Python
 ARCHITECTURE : End-to-End Data Science Lifecycle
                (Data Ingestion -> EDA -> Preprocessing -> Modeling -> Evaluation)
 LIBRARIES    : pandas, numpy, scikit-learn, matplotlib, seaborn
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless mode for saving charts cleanly
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------------------------
# GLOBAL CONFIGURATION & PATH SETUP
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease.csv")
PLOT_DIR = os.path.join(BASE_DIR, "output_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Set visual styling for all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# ===========================================================================
# STEP 1: DATA INGESTION & EXPLORATORY SUMMARY
# ===========================================================================
def load_and_inspect_data(filepath):
    """
    Loads dataset from CSV, performs structural inspection and integrity checks.
    Returns: pandas DataFrame
    """
    print("=" * 70)
    print(" STEP 1: DATA INGESTION & DATASET INTEGRITY CHECK")
    print("=" * 70)

    if not os.path.exists(filepath):
        print(f"[ERROR] Dataset not found at: {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"[OK] Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    print("--- First 5 Records ---")
    print(df.head())

    print("\n--- Dataset Schema & Data Types ---")
    df.info()

    print("\n--- Missing / Null Values Check ---")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "[OK] No missing values detected!")

    print("\n--- Descriptive Statistical Summary ---")
    print(df.describe().round(2))

    print("\n--- Target Class Distribution (0 = Healthy, 1 = Heart Disease) ---")
    target_counts = df['target'].value_counts()
    print(f"  Class 0 (Healthy / No Disease) : {target_counts.get(0, 0)} ({target_counts.get(0, 0) / len(df) * 100:.1f}%)")
    print(f"  Class 1 (Heart Disease Present): {target_counts.get(1, 0)} ({target_counts.get(1, 0) / len(df) * 100:.1f}%)")
    print("=" * 70)
    return df


# ===========================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
# ===========================================================================
def perform_eda(df):
    """
    Generates exploratory charts to analyze feature distributions and correlations.
    Saves all figures into output_plots/.
    """
    print("\n" + "=" * 70)
    print(" STEP 2: EXPLORATORY DATA ANALYSIS & VISUALIZATIONS")
    print("=" * 70)

    # Plot 1: Target Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['target'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(['Healthy (0)', 'Heart Disease (1)'], [counts[0], counts[1]], color=colors, width=0.55, edgecolor='black', alpha=0.85)
    ax.set_title("Target Distribution: Healthy vs Heart Disease Patients", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Number of Patients", fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 3, f"{yval} ({yval/len(df)*100:.1f}%)",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plot1_path = os.path.join(PLOT_DIR, "01_target_distribution.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot1_path}")

    # Plot 2: Age Distribution by Heart Disease Status
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='age', hue='target', kde=True, bins=20,
                 palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.6, ax=ax)
    ax.set_title("Patient Age Distribution by Heart Disease Status", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Age (Years)", fontsize=11)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.legend(labels=['Heart Disease (1)', 'Healthy (0)'], title="Condition")
    plt.tight_layout()
    plot2_path = os.path.join(PLOT_DIR, "02_age_vs_disease.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot2_path}")

    # Plot 3: Feature Correlation Heatmap
    fig, ax = plt.subplots(figsize=(11, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, linecolor='#ffffff', cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
    ax.set_title("Medical Features Correlation Matrix", fontsize=14, fontweight='bold', pad=14)
    plt.tight_layout()
    plot3_path = os.path.join(PLOT_DIR, "03_correlation_heatmap.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot3_path}")

    # Plot 4: Cholesterol vs Resting Blood Pressure
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='trestbps', y='chol', hue='target',
                    palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.85, s=70, ax=ax)
    ax.set_title("Serum Cholesterol vs Resting Blood Pressure", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Resting Blood Pressure (mm Hg)", fontsize=11)
    ax.set_ylabel("Serum Cholesterol (mg/dl)", fontsize=11)
    ax.axvline(120, color='gray', linestyle='--', alpha=0.7, label='Normal BP (<120)')
    ax.axhline(200, color='purple', linestyle='--', alpha=0.7, label='Desirable Chol (<200)')
    ax.legend(title="Status / Thresholds")
    plt.tight_layout()
    plot4_path = os.path.join(PLOT_DIR, "04_chol_vs_bp.png")
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot4_path}")

    # Plot 5: Maximum Heart Rate Achieved vs Age
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='age', y='thalach', hue='target',
                    palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.85, s=70, ax=ax)
    ax.set_title("Max Heart Rate (thalach) vs Patient Age", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Age (Years)", fontsize=11)
    ax.set_ylabel("Max Heart Rate Achieved (bpm)", fontsize=11)
    ax.legend(labels=['Heart Disease (1)', 'Healthy (0)'], title="Status")
    plt.tight_layout()
    plot5_path = os.path.join(PLOT_DIR, "05_heart_rate_vs_age.png")
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot5_path}")
    print("=" * 70)


# ===========================================================================
# STEP 3: PREPROCESSING & TRAIN-TEST SPLIT
# ===========================================================================
def preprocess_and_split(df):
    """
    Separates features (X) and label (y).
    Splits into 80% Train and 20% Test sets with stratification.
    Applies StandardScaler to normalize feature ranges.
    Returns: X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    print("\n" + "=" * 70)
    print(" STEP 3: DATA PREPROCESSING & STRATIFIED SPLIT")
    print("=" * 70)

    # 1. Feature Matrix (X) and Target Vector (y)
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()

    # 2. Train-Test Split (80% Train, 20% Test)
    # Stratify ensures exact target class ratio is preserved across both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"  Total samples      : {len(df)}")
    print(f"  Training samples   : {len(X_train)} (80%)")
    print(f"  Testing samples    : {len(X_test)} (20%)")

    # 3. Feature Scaling using StandardScaler
    # Note: Fit ONLY on training data to prevent Data Leakage!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("  [OK] Features standardized using StandardScaler (mean=0, variance=1).")
    print("  [VIVA NOTE] Scaler was fitted strictly on X_train to prevent data leakage.")
    print("=" * 70)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ===========================================================================
# STEP 4: MODEL TRAINING & MULTI-ALGORITHM BENCHMARK
# ===========================================================================
def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Trains 4 fundamental syllabus algorithms:
      1. Logistic Regression
      2. Decision Tree Classifier
      3. Random Forest Classifier
      4. K-Nearest Neighbors (KNN)
    Evaluates each using Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    print("\n" + "=" * 70)
    print(" STEP 4: MACHINE LEARNING MODEL TRAINING & BENCHMARKING")
    print("=" * 70)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    results = []
    trained_models = {}
    conf_matrices = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predict on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Compute performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        conf_matrices[name] = cm

        results.append({
            "Algorithm": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        print(f"\n--- {name} ---")
        print(f"  Accuracy  : {acc * 100:.2f}%")
        print(f"  Precision : {prec * 100:.2f}%")
        print(f"  Recall    : {rec * 100:.2f}%")
        print(f"  F1-Score  : {f1 * 100:.2f}%")
        print(f"  ROC-AUC   : {auc * 100:.2f}%")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Healthy', 'Heart Disease'], digits=3))

    results_df = pd.DataFrame(results)

    print("=" * 70)
    print(" SUMMARY COMPARISON TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Save Model Comparison Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(results_df['Algorithm'], results_df['Accuracy'] * 100,
                  color=['#3498db', '#e67e22', '#2ecc71', '#9b59b6'], width=0.5, edgecolor='black', alpha=0.9)
    ax.set_ylim([60, 100])
    ax.set_ylabel("Accuracy Score (%)", fontsize=11)
    ax.set_title("Machine Learning Models Accuracy Comparison", fontsize=13, fontweight='bold', pad=12)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plot6_path = os.path.join(PLOT_DIR, "06_model_accuracy_comparison.png")
    plt.savefig(plot6_path, dpi=300)
    plt.close()
    print(f"\n  [SAVED] {plot6_path}")

    # Save Confusion Matrices Side-by-Side
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for idx, (name, cm) in enumerate(conf_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Pred Healthy', 'Pred Disease'],
                    yticklabels=['True Healthy', 'True Disease'])
        axes[idx].set_title(f"{name}", fontsize=11, fontweight='bold')
    plt.suptitle("Confusion Matrices Across Models", fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plot7_path = os.path.join(PLOT_DIR, "07_confusion_matrices.png")
    plt.savefig(plot7_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot7_path}")

    # Random Forest Feature Importance Analysis
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False, ax=ax, edgecolor='black')
    ax.set_title("Clinical Feature Importance (Random Forest)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Relative Importance Weight", fontsize=11)
    plt.tight_layout()
    plot8_path = os.path.join(PLOT_DIR, "08_feature_importance.png")
    plt.savefig(plot8_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot8_path}")

    return trained_models, results_df


# ===========================================================================
# STEP 5: PATIENT RISK PREDICTION UTILITY
# ===========================================================================
def predict_patient_risk(model, scaler, patient_data, feature_names):
    """
    Demonstrates model deployment: takes raw patient metrics, applies scaling,
    and returns predicted status, probability percentage, and medical advisory.
    """
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    patient_scaled = scaler.transform(patient_df)

    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1] * 100

    return prediction, probability


def run_demo_predictions(trained_models, scaler, feature_names):
    """Runs demonstration of patient predictions on two clinical profiles."""
    print("\n" + "=" * 70)
    print(" STEP 5: CLINICAL PREDICTION DEMONSTRATION")
    print("=" * 70)

    # Use Random Forest / Logistic Regression for demonstration
    best_model = trained_models["Logistic Regression"]

    # Sample Patient A: Healthy Profile (38 yo, BP 115, Chol 180, MaxHR 175, 0 vessels blocked, normal ST)
    patient_a = {
        'age': 38, 'sex': 0, 'cp': 2, 'trestbps': 115, 'chol': 180,
        'fbs': 0, 'restecg': 0, 'thalach': 175, 'exang': 0, 'oldpeak': 0.2,
        'slope': 1, 'ca': 0, 'thal': 3
    }

    # Sample Patient B: High-Risk Profile (67 yo male, BP 160, Chol 286, MaxHR 108, ST depression 2.6, 2 blocked vessels)
    patient_b = {
        'age': 67, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 286,
        'fbs': 0, 'restecg': 2, 'thalach': 108, 'exang': 1, 'oldpeak': 2.6,
        'slope': 2, 'ca': 2, 'thal': 7
    }

    for name, patient in [("Patient A (Healthy Profile)", patient_a), ("Patient B (High-Risk Profile)", patient_b)]:
        pred, prob = predict_patient_risk(best_model, scaler, patient, feature_names)
        status_label = "HIGH RISK (Heart Disease Detected)" if pred == 1 else "LOW RISK (Healthy / No Disease)"
        print(f"\n--- {name} ---")
        print(f"  Clinical Inputs: Age={patient['age']}, BP={patient['trestbps']} mmHg, Chol={patient['chol']} mg/dL, MaxHR={patient['thalach']} bpm, Blocked Vessels={patient['ca']}")
        print(f"  Prediction     : {status_label}")
        print(f"  Risk Probability: {prob:.1f}%")
        if pred == 1:
            print("  Advisory       : Elevated cardiovascular risk markers detected. Cardiologist consultation recommended.")
        else:
            print("  Advisory       : Cardiovascular readings within normal limits. Continue healthy diet and regular exercise.")
    print("=" * 70)


# ===========================================================================
# MAIN PIPELINE EXECUTION
# ===========================================================================
def main():
    print("""
    ================================================================
            HEART ATTACK ANALYSIS (DATA SCIENCE USING PYTHON)
    ================================================================
    """)
    # 1. Ingestion
    df = load_and_inspect_data(DATA_PATH)

    # 2. Exploratory Data Analysis
    perform_eda(df)

    # 3. Preprocessing & Splitting
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_and_split(df)

    # 4. Model Training & Benchmarking
    trained_models, results_df = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )

    # 5. Live Demonstration
    run_demo_predictions(trained_models, scaler, feature_names)

    print("\n[SUCCESS] Project pipeline finished executing cleanly.")
    print(f"[OUTPUT] All visualization charts saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
