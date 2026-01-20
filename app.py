import pandas as pd
import joblib
from flask import Flask, request, jsonify
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Load the trained model ---
model_file_path = 'random_forest_model.joblib'
if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"Model file not found at {model_file_path}. Please ensure the model is saved correctly.")

model = joblib.load(model_file_path)

# --- Define the exact feature names and categorical mapping from training ---
# These lists are crucial to ensure consistency between training and inference
# Numerical columns that are directly passed
numerical_features = [
    'Age', 'Daily_Screen_Time_Hours', 'Late_Night_Usage', 
    'Social_Comparison_Trigger', 'Sleep_Duration_Hours', 
    'GAD_7_Score', 'PHQ_9_Score'
]

# Original categorical columns and all unique values observed during training
# This is used to ensure consistent one-hot encoding.
original_categorical_columns_mapping = {
    'Gender': ['Female', 'Male'],
    'User_Archetype': ['Digital Minimalist', 'Hyper-Connected', 'Tech-Savvy', 'Traditionalist'],
    'Primary_Platform': ['Discord', 'Facebook', 'Instagram', 'LinkedIn', 'Pinterest', 'Reddit', 'Snapchat', 'TikTok', 'Twitter/X', 'YouTube'],
    'Dominant_Content_Type': ['Educational/Informative', 'Entertainment/Comedy', 'Gaming', 'Lifestyle/Fashion', 'News/Politics', 'Self-Help/Motivation'],
    'Activity_Type': ['Active', 'Passive'],
    'GAD_7_Severity': ['Mild', 'Minimal', 'Moderate', 'Severely Severe', 'Severe']
}

# The exact list of feature columns the model was trained on, in the correct order.
# This list is reconstructed based on the `X.shape` and `X.head()` output from the notebook.
model_expected_features = numerical_features + [
    'Gender_Male',
    'User_Archetype_Digital Minimalist', 'User_Archetype_Hyper-Connected', 'User_Archetype_Tech-Savvy',
    'Primary_Platform_Pinterest', 'Primary_Platform_Reddit', 'Primary_Platform_Snapchat',
    'Primary_Platform_TikTok', 'Primary_Platform_Twitter/X', 'Primary_Platform_YouTube',
    'Dominant_Content_Type_Entertainment/Comedy', 'Dominant_Content_Type_Gaming',
    'Dominant_Content_Type_Lifestyle/Fashion', 'Dominant_Content_Type_News/Politics',
    'Dominant_Content_Type_Self-Help/Motivation',
    'Activity_Type_Passive',
    'GAD_7_Severity_Minimal', 'GAD_7_Severity_Moderate', 'GAD_7_Severity_Severe'
]

# --- Preprocessing function for API input ---
def preprocess_input(input_data: dict) -> pd.DataFrame:
    # Create a DataFrame from the input data, handling missing keys
    processed_df = pd.DataFrame([input_data])

    # Handle numerical features: ensure they are present and of correct type
    for col in numerical_features:
        if col not in processed_df.columns or processed_df[col].isnull().any():
            # You might want to replace NaN with a default/median/mean value
            # For now, we'll raise an error or fill with 0, depending on expected behavior.
            # For simplicity, filling with 0 here, but in a real scenario, use training means/medians.
            processed_df[col] = processed_df.get(col, 0.0).fillna(0.0).astype(float)
        else:
            processed_df[col] = processed_df[col].astype(float)

    # Handle categorical features: one-hot encode them consistently
    for col, categories in original_categorical_columns_mapping.items():
        if col not in processed_df.columns:
            processed_df[col] = None # Add column if missing
        # Ensure the column is of 'category' dtype with all known categories
        processed_df[col] = pd.Categorical(processed_df[col], categories=categories)

    # Apply one-hot encoding with drop_first=True, exactly as in training
    processed_df = pd.get_dummies(processed_df, 
                                  columns=list(original_categorical_columns_mapping.keys()), 
                                  drop_first=True, dtype=int)

    # Reindex the DataFrame to match the model's expected feature order and presence
    # Fill missing columns with 0 (for categories not present in the input) and drop extra ones.
    final_features_df = processed_df.reindex(columns=model_expected_features, fill_value=0)

    # Ensure boolean columns (from get_dummies) are converted to int (0 or 1)
    for col in final_features_df.columns:
        if final_features_df[col].dtype == 'bool':
            final_features_df[col] = final_features_df[col].astype(int)

    return final_features_df

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    
    # Basic validation for required numerical features
    required_numerical = set(numerical_features)
    if not required_numerical.issubset(data.keys()):
        missing = required_numerical - set(data.keys())
        return jsonify({'error': f'Missing required numerical features: {list(missing)}'}), 400
    
    try:
        # Preprocess the input data
        processed_data = preprocess_input(data)
        
        # Ensure the processed data has the correct number of columns before prediction
        if processed_data.shape[1] != len(model_expected_features):
            return jsonify({'error': f'Processed data has {processed_data.shape[1]} features, but model expects {len(model_expected_features)}. Column mismatch occurred.'}), 400

        # Make prediction
        prediction = model.predict(processed_data)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]}), 200
    except KeyError as e:
        return jsonify({'error': f'Key error in input data: {e}. Please ensure all expected fields are present and correctly named.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Run the Flask app ---
# In Colab, you might need to run this on '0.0.0.0' to be accessible.
# For local testing, you can use `app.run(debug=True)`

# To run the app, execute this cell and then send POST requests to:
# http://<your_colab_ip>:5000/predict
# You might need a tool like ngrok to expose your Colab port to the internet
# if you want to access it from outside Colab.
print("Flask API setup complete. To run the server, execute this cell.")
print("You can then make POST requests to /predict with JSON input.")

# For demonstration in Colab, we'll keep the app running in the cell.
# In a real deployment, you'd use a production-ready WSGI server like Gunicorn.
# Uncomment the following line to run the app:
# app.run(host='0.0.0.0', port=5000, debug=False)
