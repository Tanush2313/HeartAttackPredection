"""
=============================================================================
 HEART ATTACK ANALYSIS - Predictive Analytics & Clinical Dashboard
 Backend Server: Python Flask Framework
 Subject       : Data Science Using Python
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease.csv")
PLOT_DIR = os.path.join(BASE_DIR, "output_plots")

# ---------------------------------------------------------------------------
# GLOBAL ML STATE INITIALIZATION
# ---------------------------------------------------------------------------
ml_state = {
    "df": None,
    "scaler": None,
    "feature_names": [],
    "models": {},
    "metrics": {},
    "feature_importance": [],
    "confusion_matrices": {}
}


def initialize_data_science_models():
    """Initializes dataset, trains 4 syllabus algorithms, computes benchmark metrics."""
    global ml_state
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset missing at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    ml_state["df"] = df

    X = df.drop("target", axis=1)
    y = df["target"]
    ml_state["feature_names"] = X.columns.tolist()

    # Stratified split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Feature scaling (fit only on X_train to prevent leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ml_state["scaler"] = scaler

    # Initialize 4 standard syllabus models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42)
    }

    metrics = {}
    conf_matrices = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        metrics[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "roc_auc": round(auc * 100, 2)
        }

        conf_matrices[name] = {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1])
        }

    ml_state["models"] = models
    ml_state["metrics"] = metrics
    ml_state["confusion_matrices"] = conf_matrices

    # Feature importance from Random Forest
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    fi = [
        {"feature": feat, "importance": round(float(imp) * 100, 2)}
        for feat, imp in zip(ml_state["feature_names"], importances)
    ]
    fi = sorted(fi, key=lambda x: x["importance"], reverse=True)
    ml_state["feature_importance"] = fi

    print("[SERVER INIT] Data Science models and metrics loaded successfully.")


# Initialize on startup
initialize_data_science_models()


# ---------------------------------------------------------------------------
# WEB ROUTES & REST APIS
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Renders the main analytics dashboard."""
    return render_template("index.html")


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    """Serves generated EDA and benchmark plot images."""
    return send_from_directory(PLOT_DIR, filename)


@app.route("/api/overview", methods=["GET"])
def get_overview():
    """Returns dataset summary stats, model benchmark comparison, and feature importance."""
    df = ml_state["df"]
    target_counts = df["target"].value_counts().to_dict()

    return jsonify({
        "dataset": {
            "total_records": len(df),
            "total_features": len(ml_state["feature_names"]),
            "healthy_count": int(target_counts.get(0, 0)),
            "disease_count": int(target_counts.get(1, 0)),
            "healthy_percentage": round(target_counts.get(0, 0) / len(df) * 100, 1),
            "disease_percentage": round(target_counts.get(1, 0) / len(df) * 100, 1),
            "features": ml_state["feature_names"]
        },
        "metrics": ml_state["metrics"],
        "confusion_matrices": ml_state["confusion_matrices"],
        "feature_importance": ml_state["feature_importance"]
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Receives JSON patient payload, runs standardization & ML inference.
    Returns: classification, probability %, risk tier, and medical recommendation.
    """
    try:
        data = request.get_json(force=True)
        model_name = data.get("model", "Random Forest")
        if model_name not in ml_state["models"]:
            model_name = "Random Forest"

        patient_input = {
            "age": float(data.get("age", 50)),
            "sex": int(data.get("sex", 1)),
            "cp": int(data.get("cp", 1)),
            "trestbps": float(data.get("trestbps", 120)),
            "chol": float(data.get("chol", 200)),
            "fbs": int(data.get("fbs", 0)),
            "restecg": int(data.get("restecg", 0)),
            "thalach": float(data.get("thalach", 150)),
            "exang": int(data.get("exang", 0)),
            "oldpeak": float(data.get("oldpeak", 0.0)),
            "slope": int(data.get("slope", 1)),
            "ca": int(data.get("ca", 0)),
            "thal": int(data.get("thal", 3))
        }

        # Build DataFrame with proper feature ordering
        df_patient = pd.DataFrame([patient_input], columns=ml_state["feature_names"])
        scaled_patient = ml_state["scaler"].transform(df_patient)

        model = ml_state["models"][model_name]
        prediction = int(model.predict(scaled_patient)[0])
        probability = float(model.predict_proba(scaled_patient)[0][1] * 100)

        # Risk categorization
        if probability < 30:
            tier = "Low Risk"
            badge_class = "badge-low"
            advice = "Patient clinical indicators are within healthy normative ranges. Maintain active lifestyle, healthy nutrition, and regular annual checkups."
        elif probability < 65:
            tier = "Moderate Risk"
            badge_class = "badge-moderate"
            advice = "Borderline cardiovascular indicators detected. Recommend dietary moderation, blood pressure monitoring, and lipid profile re-check within 3 months."
        else:
            tier = "High Critical Risk"
            badge_class = "badge-high"
            advice = "Significant cardiac risk markers detected. Urgent referral to a cardiologist for formal ECG treadmill stress testing and coronary angiography advised."

        return jsonify({
            "success": True,
            "model_used": model_name,
            "prediction": prediction,
            "status": "HEART DISEASE DETECTED" if prediction == 1 else "HEALTHY / NEGATIVE",
            "probability": round(probability, 1),
            "risk_tier": tier,
            "badge_class": badge_class,
            "clinical_advice": advice,
            "patient_summary": {
                "age": int(patient_input["age"]),
                "sex": "Male" if patient_input["sex"] == 1 else "Female",
                "trestbps": f"{int(patient_input['trestbps'])} mmHg",
                "chol": f"{int(patient_input['chol'])} mg/dL",
                "thalach": f"{int(patient_input['thalach'])} bpm",
                "oldpeak": patient_input["oldpeak"],
                "ca": int(patient_input["ca"])
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    # Run on port 5050 to avoid macOS Monterey/Ventura AirPlay collisions on 5000
    print("=" * 65)
    print("  HEART ATTACK ANALYSIS DASHBOARD SERVER ACTIVE")
    print("  URL: http://127.0.0.1:5050")
    print("=" * 65)
    app.run(host="0.0.0.0", port=5050, debug=False)
