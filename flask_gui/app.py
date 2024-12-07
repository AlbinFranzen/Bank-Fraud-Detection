# from flask import Flask, render_template, request
# import joblib  # To load the trained machine learning model

# app = Flask(__name__)

# # Load the pre-trained machine learning model
# model = joblib.load("../best_model.joblib")

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     if request.method == "POST":
#         try:
#             # List of expected feature names (as per the HTML form)
#             feature_names = [
#                 "income", "name_email_similarity", "prev_address_months_count",
#                 "current_address_months_count", "customer_age", "days_since_request",
#                 "intended_balcon_amount", "payment_type", "zip_count_4w", "velocity_6h",
#                 "velocity_24h", "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w", 
#                 "employment_status", "credit_risk_score", "email_is_free", "housing_status", 
#                 "phone_home_valid", "phone_mobile_valid", "bank_months_count", "has_other_cards", 
#                 "proposed_credit_limit", "foreign_request", "source", "session_length_in_minutes", 
#                 "device_os", "keep_alive_session", "device_distinct_emails_8w", "device_fraud_count", 
#                 "month"
#             ]

#             # Initialize list for feature values
#             features = []

#             # Iterate over each feature name, retrieve form value, and convert to float
#             for name in feature_names:
#                 value = request.form.get(name)
#                 if value is None or value.strip() == '':
#                     return f"Missing or empty input for field: {name}"
#                 try:
#                     features.append(float(value))
#                 except ValueError:
#                     return f"Invalid input for field '{name}'. Please enter a valid number."

#             # Reshape data for prediction and make prediction
#             prediction = model.predict([features])
#             predicted_class = prediction[0]

#             return render_template("result.html", prediction=predicted_class)

#         except Exception as e:
#             return f"An error occurred: {str(e)}"

#     return "Something went wrong. Please try again."

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import joblib  # To load the trained machine learning model
import pandas as pd  # Import pandas for DataFrame handling
from preprocessing import preprocessor  # Ensure this file contains the preprocessor pipeline

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load("../best_model.joblib")

@app.route("/")
def index():
    return render_template("index.html")

def convert_to_number (value):
                try:
                    return int(value)
                except ValueError:
                    try:
                        return float(value)
                    except ValueError:
                        raise ValueError (f"e")
                    
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # List of expected feature names (must match the form fields)
            feature_names = [
                "income", "name_email_similarity", "prev_address_months_count",
                "current_address_months_count", "customer_age", "days_since_request",
                "intended_balcon_amount", "payment_type", "zip_count_4w", "velocity_6h",
                "velocity_24h", "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w", 
                "employment_status", "credit_risk_score", "email_is_free", "housing_status", 
                "phone_home_valid", "phone_mobile_valid", "bank_months_count", "has_other_cards", 
                "proposed_credit_limit", "foreign_request", "source", "session_length_in_minutes", 
                "device_os", "keep_alive_session", "device_distinct_emails_8w", "device_fraud_count", 
                "month"
            ]
    
                    
            # Retrieve and process form data
            input_data = {}
            for name in feature_names:
                value = request.form.get(name)
                if value is None or value.strip() == '':
                    return f"Missing or empty input for field: {name}"
                try:
                    input_data[name] = convert_to_number(value) if name not in [ "payment_type", "employment_status", 
                                                                   "housing_status", "device_os", "source", "month"] else value
                except ValueError:
                    return f"Invalid input for field '{name}'. Please enter a valid number."

            

            # Convert the dictionary into a DataFrame
            print(input_data)
            input_df = pd.DataFrame([input_data])  # Single row DataFrame
            print (input_df)
            # Preprocess the data
            total_df = pd.read_csv('./Data/Base.csv')
            X = total_df.drop(columns=['fraud_bool'])
            preprocessor.fit(X)  # Use transform, not fit_transform
            print (input_df.iloc[:, 0])
            preprocessed_data = preprocessor.transform (input_df)
            

            # Predict using the pre-trained model
            prediction = model.predict(preprocessed_data)
            predicted_class = prediction[0]  # Get the first prediction

            # Render the result
            return render_template("result.html", prediction=predicted_class)

        except Exception as e:
            return f"An error occurred: {str(e)}"

    return "Something went wrong. Please try again."

if __name__ == "__main__":
    app.run(debug=True)
