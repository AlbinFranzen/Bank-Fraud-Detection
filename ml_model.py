# ml_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import joblib

# Generate dummy data with 33 features to match the form inputs
X, y = make_classification(
    n_samples=1000,             # Increase the number of samples for better training
    n_features=32,              # 33 features as per the HTML form
    n_informative=20,           # Number of informative features
    n_redundant=5,              # Some redundant features to simulate noise
    n_repeated=0,               # No repeated features
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter to ensure convergence with more features
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, "classification_model.joblib")
