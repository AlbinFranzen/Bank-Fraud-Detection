from flask import Flask, render_template, request
import joblib  # To load the trained machine learning model
import jsonify
import pandas as pd
import numpy as np
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
model = joblib.load("./flask_gui/gradb.joblib")

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



@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Convert and encode features
            error, input_df = convert_features(request, feature_types)
            
            if error:
                return jsonify({"error": error}), 400  # Return a JSON response with the error
            
            # Make prediction using your trained model
            print(model)
            processed_input = preprocess_pipeline.transform(input_df)
            prediction = model.predict(processed_input)
            predicted_class = prediction[0]
            
            # Optionally, map the predicted class to a human-readable label
            # For example:
            # class_mapping = {0: "Not Fraud", 1: "Fraud"}
            # predicted_label = class_mapping.get(predicted_class, "Unknown")
            
            return render_template("result.html", prediction=predicted_class)  # Or use jsonify if API
            # Example using jsonify:
            # return jsonify({"prediction": predicted_class}), 200
        
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500  # Return a JSON response with the error
    
    return jsonify({"error": "Invalid request method."}), 405  # Method Not Allowed


if __name__ == "__main__":
    app.run(debug=True)
