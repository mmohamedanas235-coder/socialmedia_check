import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# =========================
# Flask App Initialization
# =========================
app = Flask(__name__)

# =========================
# Load Model & Scaler (FIXED NAMES)
# =========================
MODEL_PATH = "random_forest_model.joblib"
SCALER_PATH = "scaler (1).joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =========================
# Feature Definitions
# =========================
numerical_features = [
    "Age",
    "Daily_Screen_Time_Hours",
    "Late_Night_Usage",
    "Social_Comparison_Trigger",
    "Sleep_Duration_Hours",
    "GAD_7_Score",
    "PHQ_9_Score",
]

categorical_mapping = {
    "Gender": ["Female", "Male"],
    "User_Archetype": [
        "Digital Minimalist",
        "Hyper-Connected",
        "Tech-Savvy",
        "Traditionalist",
    ],
    "Primary_Platform": [
        "Discord",
        "Facebook",
        "Instagram",
        "LinkedIn",
        "Pinterest",
        "Reddit",
        "Snapchat",
        "TikTok",
        "Twitter/X",
        "YouTube",
    ],
    "Dominant_Content_Type": [
        "Educational/Informative",
        "Entertainment/Comedy",
        "Gaming",
        "Lifestyle/Fashion",
        "News/Politics",
        "Self-Help/Motivation",
    ],
    "Activity_Type": ["Active", "Passive"],
    "GAD_7_Severity": [
        "Mild",
        "Minimal",
        "Moderate",
        "Severely Severe",
        "Severe",
    ],
}

model_expected_features = numerical_features + [
    "Gender_Male",
    "User_Archetype_Digital Minimalist",
    "User_Archetype_Hyper-Connected",
    "User_Archetype_Tech-Savvy",
    "Primary_Platform_Pinterest",
    "Primary_Platform_Reddit",
    "Primary_Platform_Snapchat",
    "Primary_Platform_TikTok",
    "Primary_Platform_Twitter/X",
    "Primary_Platform_YouTube",
    "Dominant_Content_Type_Entertainment/Comedy",
    "Dominant_Content_Type_Gaming",
    "Dominant_Content_Type_Lifestyle/Fashion",
    "Dominant_Content_Type_News/Politics",
    "Dominant_Content_Type_Self-Help/Motivation",
    "Activity_Type_Passive",
    "GAD_7_Severity_Minimal",
    "GAD_7_Severity_Moderate",
    "GAD_7_Severity_Severe",
]

# =========================
# Preprocessing Function
# =========================
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Numerical preprocessing
    for col in numerical_features:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    df[numerical_features] = scaler.transform(df[numerical_features])

    # Categorical preprocessing
    for col, categories in categorical_mapping.items():
        df[col] = pd.Categorical(
            df.get(col, None),
            categories=categories
        )

    df = pd.get_dummies(
        df,
        columns=categorical_mapping.keys(),
        drop_first=True,
        dtype=int,
    )

    # Align columns exactly to training
    df = df.reindex(columns=model_expected_features, fill_value=0)

    return df

# =========================
# Prediction Endpoint
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    missing = set(numerical_features) - data.keys()
    if missing:
        return jsonify({
            "error": f"Missing required numerical features: {list(missing)}"
        }), 400

    try:
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)

        return jsonify({"prediction": prediction[0]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# Run Server
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
