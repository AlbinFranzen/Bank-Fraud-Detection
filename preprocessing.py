import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer, OneHotEncoder
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

### NUMERICAL PREPROCESSOR
# Define the ColumnTransformer pipeline with drop columns integrated
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
    # Drop specified columns directly in the numerical preprocessor
    ('drop_columns', FunctionTransformer(drop_columns, validate=False))
])

### BOOLEAN PREPROCESSOR
# Define transformation functions for each feature in boolean preprocessing
def total_valid_phones(X):
    return (X['phone_home_valid'] + X['phone_mobile_valid']).values.reshape(-1, 1)

def foreign_long_session(X):
    return (X['foreign_request'] & X['keep_alive_session']).astype(int).values.reshape(-1, 1)

def device_and_account_history(X):
    return (X['device_fraud_count'] * X['keep_alive_session']).values.reshape(-1, 1)

def total_risk_flags(X):
    high_risk_features = ['email_is_free', 'foreign_request', 'has_other_cards', 'keep_alive_session']
    return X[high_risk_features].sum(axis=1).values.reshape(-1, 1)

# Define the ColumnTransformer for boolean feature engineering
boolean_preprocessor = ColumnTransformer(
    transformers=[
        ('total_valid_phones', FunctionTransformer(total_valid_phones), ['phone_home_valid', 'phone_mobile_valid']),
        ('foreign_long_session', FunctionTransformer(foreign_long_session), ['foreign_request', 'keep_alive_session']),
        ('device_and_account_history', FunctionTransformer(device_and_account_history), ['device_fraud_count', 'keep_alive_session']),
        ('total_risk_flags', FunctionTransformer(total_risk_flags), ['email_is_free', 'foreign_request', 'has_other_cards', 'keep_alive_session']),
        ('source_encoded', 'passthrough', ['source'])  # Assuming 'source' is preprocessed externally
    ],
    remainder='passthrough'
)

### CATEGORICAL PREPROCESSOR
# Low-cardinality categorical features (One-Hot Encoding)
low_card_preprocessor = ColumnTransformer(
    transformers=[
        ('one_hot_encoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 'low_card')
    ],
    remainder='passthrough'
)

# High-cardinality categorical features (Ordinal Encoding)
high_card_preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal_encoding', OrdinalEncoder(), 'high_card')
    ],
    remainder='passthrough'
)

### COMBINED PREPROCESSOR PIPELINE
# Combined pipeline that runs numerical preprocessing, boolean preprocessing, and categorical preprocessing
preprocessor = Pipeline([
    ('numerical_processing', numerical_preprocessor),
    ('boolean_processing', boolean_preprocessor),
    ('low_card_categorical_processing', low_card_preprocessor),
    ('high_card_categorical_processing', high_card_preprocessor)
])

# Usage example:
# transformed_data = preprocessor.fit_transform(train_df)
