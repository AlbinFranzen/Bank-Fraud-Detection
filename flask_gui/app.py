from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import roc_curve, auc

def recall_at_fpr(y_true, y_scores, target_fpr):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(fpr <= target_fpr)[0][-1] if np.any(fpr <= target_fpr) else None
    return tpr[idx] if idx is not None else 0.0



# Get preprocessor
try:
    preprocess_pipeline = joblib.load('./flask_gui/preprocess_pipeline.joblib')
except FileNotFoundError:
    print("Pipeline file not found. Ensure 'preprocess_pipeline.joblib' exists.")
    preprocess_pipeline = None

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load("./flask_gui/best_xgb_model.joblib")

@app.route("/")
def index():
    return render_template("index.html")

# Define the feature types
feature_types = {
    "income": "float",
    "name_email_similarity": "float",
    "prev_address_months_count": "int",
    "current_address_months_count": "int",
    "customer_age": "int",
    "days_since_request": "float",
    "intended_balcon_amount": "float",
    "payment_type": "object",
    "zip_count_4w": "int",
    "velocity_6h": "float",
    "velocity_24h": "float",
    "velocity_4w": "float",
    "bank_branch_count_8w": "int",
    "date_of_birth_distinct_emails_4w": "int",
    "employment_status": "object",
    "credit_risk_score": "int",
    "email_is_free": "int",
    "housing_status": "object",
    "phone_home_valid": "int",
    "phone_mobile_valid": "int",
    "bank_months_count": "int",
    "has_other_cards": "int",
    "proposed_credit_limit": "float",
    "foreign_request": "int",
    "source": "object",
    "session_length_in_minutes": "float",
    "device_os": "object",
    "keep_alive_session": "int",
    "device_distinct_emails_8w": "int",
    "device_fraud_count": "int",
    "month": "int"
}

def convert_features(request, feature_types):
    """
    Convert form inputs to their respective types and return as a pandas DataFrame.

    Parameters:
    - request: Flask request object containing form data.
    - feature_types: Dictionary mapping feature names to their data types.

    Returns:
    - Tuple containing an error message (if any) and the DataFrame of features.
    """
    # Initialize a dictionary to hold feature values
    features = {}
    
    # Iterate over each feature name and its type
    for feature, f_type in feature_types.items():
        value = request.form.get(feature)
        
        # Check for missing or empty inputs
        if value is None or value.strip() == '':
            return f"Missing or empty input for field: {feature}", None
        
        try:
            if f_type == "int":
                converted_value = int(value)
            elif f_type == "float":
                converted_value = float(value)
            elif f_type == "object":
                # For categorical features, keep as string or handle encoding here
                converted_value = value.strip()
            else:
                return f"Unknown type for feature: {feature}", None
            
            features[feature] = converted_value
        
        except ValueError:
            return f"Invalid input for field '{feature}'. Please enter a valid {f_type}.", None
    
    # Create a pandas DataFrame from the features dictionary
    df = pd.DataFrame([features])
    
    return None, df



import shap
from sklearn.metrics import roc_auc_score, roc_curve
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Convert form data to DataFrame
            error, input_df = convert_features(request, feature_types)
            if error:
                return jsonify({"error": error}), 400
            
            # Preprocess input
            processed_input = preprocess_pipeline.transform(input_df)
            probabilities = model.predict_proba(processed_input)[0]
            
            # Predicted class and confidence
            predicted_class = np.argmax(probabilities)

            # Generate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_input)

            # Handle binary classification SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Generate SHAP Summary Plot
            plt.figure()  # Create a new figure for SHAP
            shap.summary_plot(shap_values, processed_input, max_display=processed_input.shape[1])

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            shap_plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
            buf.close()
            plt.close()  # Close the SHAP figure

            # Render results
            return render_template(
                "result.html", 
                prediction=predicted_class,
                shap_plot=shap_plot_url
            )
        
        except ValueError as ve:
            return jsonify({"error": f"ValueError: {str(ve)}"}), 400
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid request method."}), 405



if __name__ == "__main__":
    app.run(debug=True)
