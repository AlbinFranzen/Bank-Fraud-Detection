import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to handle outlier removal after scaling
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3.0):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[np.abs(X) < self.threshold]

# Binning and ordinal encoding for `name_email_similarity`
def bin_name_email_similarity(X):
    bins = [0, 0.24, 0.48, 0.72, 0.96, 1.0]
    return pd.cut(X, bins=bins, labels=False).values.reshape(-1, 1)

# Log transformation function
log_transformer = FunctionTransformer(np.log1p, validate=False)

# Function to drop specific columns
def drop_columns(X):
    if isinstance(X, pd.DataFrame):
        return X.drop(columns=['velocity_4w', 'session_length_in_minutes'])
    return X

# Define the ColumnTransformer pipeline
numerical_preprocessor = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            # Name Email Similarity: Binning, Ordinal Encoding
            ('name_email_similarity', Pipeline([
                ('binning', FunctionTransformer(bin_name_email_similarity, validate=False)),
                ('ordinal_encoder', OrdinalEncoder())
            ]), ['name_email_similarity']),

            # Days Since Request: Log-transform, Standard Scale, Outlier Removal
            ('days_since_request', Pipeline([
                ('log_transform', log_transformer),
                ('scaler', StandardScaler()),
                ('outlier_remover', OutlierRemover())
            ]), ['days_since_request']),

            # Intended Balcon Amount: Impute, Missing Flag, Log-transform, Standard Scale, Outlier Removal
            ('intended_balcon_amount', Pipeline([
                ('missing_flag', FunctionTransformer(lambda x: (x == -1).astype(int), validate=False)),
                ('imputer', SimpleImputer(missing_values=-1, strategy='constant', fill_value=0)),
                ('log_transform', log_transformer),
                ('scaler', StandardScaler()),
                ('outlier_remover', OutlierRemover())
            ]), ['intended_balcon_amount']),

            # Velocity Metrics: Imputations and Scaling
            ('velocity_6h', Pipeline([
                ('negative_imputer', SimpleImputer(missing_values=-1, strategy='constant', fill_value=0)),
                ('scaler', StandardScaler())
            ]), ['velocity_6h']),

            ('velocity_24h', Pipeline([
                ('scaler', StandardScaler())
            ]), ['velocity_24h'])
        ],
        remainder='passthrough',  # Keep all other columns except those explicitly specified
        verbose_feature_names_out=False  # Prevent verbose names for ease of interpretation
    )),
    # Drop specified columns
    ('drop_columns', FunctionTransformer(drop_columns, validate=False))
])

# Usage example:
# transformed_data = numerical_preprocessor.fit_transform(train_df)
