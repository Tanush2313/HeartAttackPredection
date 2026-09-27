"""
=============================================================================
 INTERACTIVE CLINICAL PREDICTOR TOOL
 SUBJECT : Data Science Using Python
 USE CASE: Rapid Patient Risk Assessment & Live Examiner Demo
=============================================================================
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease.csv")


def load_model_and_scaler():
    """Trains the production Random Forest model on the dataset and returns model + scaler."""
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, feature_names


def assess_risk(model, scaler, feature_names, patient_data):
    """Computes risk probability and clinical assessment."""
    df_patient = pd.DataFrame([patient_data], columns=feature_names)
    scaled_patient = scaler.transform(df_patient)

    pred = model.predict(scaled_patient)[0]
    prob = model.predict_proba(scaled_patient)[0][1] * 100

    print("\n" + "=" * 60)
    print("        CLINICAL RISK ASSESSMENT REPORT")
    print("=" * 60)
    print(f" Patient Age               : {patient_data['age']} years")
    print(f" Gender                    : {'Male' if patient_data['sex'] == 1 else 'Female'}")
    print(f" Resting Blood Pressure    : {patient_data['trestbps']} mm Hg")
    print(f" Serum Cholesterol         : {patient_data['chol']} mg/dL")
    print(f" Max Heart Rate Achieved   : {patient_data['thalach']} bpm")
    print(f" Exercise Angina           : {'Present' if patient_data['exang'] == 1 else 'Absent'}")
    print(f" ST Depression (Oldpeak)   : {patient_data['oldpeak']}")
    print(f" Major Blocked Vessels (ca): {patient_data['ca']}")
    print("-" * 60)

    if prob < 35:
        tier = "LOW RISK"
        color_code = "[GREEN]"
        recommendation = "Metrics are within healthy ranges. Maintain an active lifestyle and balanced diet."
    elif prob < 65:
        tier = "MODERATE RISK"
        color_code = "[YELLOW]"
        recommendation = "Borderline cardiovascular indicators. Schedule routine monitoring and cholesterol screening."
    else:
        tier = "HIGH RISK"
        color_code = "[RED]"
        recommendation = "Significant cardiac risk patterns identified. Comprehensive cardiologist evaluation advised."

    print(f" Diagnostic Prediction    : {'HEART DISEASE DETECTED' if pred == 1 else 'HEALTHY / NEGATIVE'}")
    print(f" Estimated Cardiac Risk   : {prob:.1f}% ({tier} {color_code})")
    print(f" Medical Recommendation   : {recommendation}")
    print("=" * 60 + "\n")


def main():
    print("""
    +-----------------------------------------------------------+
    |   HEART ATTACK ANALYSIS - INTERACTIVE CLINICAL TESTER     |
    +-----------------------------------------------------------+
    """)
    model, scaler, feature_names = load_model_and_scaler()

    preset_healthy = {
        'age': 38, 'sex': 0, 'cp': 2, 'trestbps': 115, 'chol': 180,
        'fbs': 0, 'restecg': 0, 'thalach': 175, 'exang': 0, 'oldpeak': 0.2,
        'slope': 1, 'ca': 0, 'thal': 3
    }

    preset_high_risk = {
        'age': 67, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 286,
        'fbs': 0, 'restecg': 2, 'thalach': 108, 'exang': 1, 'oldpeak': 2.6,
        'slope': 2, 'ca': 2, 'thal': 7
    }

    preset_moderate_risk = {
        'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 135, 'chol': 245,
        'fbs': 0, 'restecg': 1, 'thalach': 145, 'exang': 0, 'oldpeak': 1.0,
        'slope': 2, 'ca': 1, 'thal': 3
    }

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if "healthy" in arg:
            assess_risk(model, scaler, feature_names, preset_healthy)
            return
        elif "risk" in arg or "high" in arg:
            assess_risk(model, scaler, feature_names, preset_high_risk)
            return
        elif "mod" in arg:
            assess_risk(model, scaler, feature_names, preset_moderate_risk)
            return

    print("Select a test option:")
    print("  1. Test Preset Healthy Patient Profile")
    print("  2. Test Preset High-Risk Cardiac Patient Profile")
    print("  3. Test Preset Moderate-Risk Patient Profile")
    print("  4. Exit")

    try:
        choice = input("\nEnter choice [1-4]: ").strip()
        if choice == '1':
            assess_risk(model, scaler, feature_names, preset_healthy)
        elif choice == '2':
            assess_risk(model, scaler, feature_names, preset_high_risk)
        elif choice == '3':
            assess_risk(model, scaler, feature_names, preset_moderate_risk)
        else:
            print("Exiting interactive predictor.")
    except (EOFError, KeyboardInterrupt):
        # Graceful fallback when run in non-interactive terminal
        print("\nRunning default demonstration (Preset Healthy vs Preset High Risk):")
        assess_risk(model, scaler, feature_names, preset_healthy)
        assess_risk(model, scaler, feature_names, preset_high_risk)


if __name__ == "__main__":
    main()
