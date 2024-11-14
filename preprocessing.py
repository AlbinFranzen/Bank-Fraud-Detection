# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, chi2, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.under_sampling import NearMiss
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from preprocessing_low_card import LowCardinalityPreprocessor
from sklearn.model_selection import train_test_split

# Preprocessing Pipelines
# =======================

# 1. Numerical Feature Engineering 
class NumericalTransformer(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Drop specified columns
        X.drop(columns=['velocity_4w', 'session_length_in_minutes'], inplace=True)
        
        # Process 'name_email_similarity'
        X['name_email_similarity_binned'] = pd.cut( X['name_email_similarity'],
            bins=[0.0, 0.24, 0.48, 0.72, 0.96, 1.0],
            labels=[0, 1, 2, 3, 4]).astype(int)
        X.drop(columns=['name_email_similarity'], inplace=True)
        
        # Log transform 'days_since_request'
        X['days_since_request'] = np.log1p(X['days_since_request'])
        
        # Process 'intended_balcon_amount'
        X['intended_balcon_amount_flag'] = (X['intended_balcon_amount'] == -1).astype(int)
        X['intended_balcon_amount'] = X['intended_balcon_amount'].clip(lower=0)
        X['intended_balcon_amount'] = np.log1p(X['intended_balcon_amount'])
        
        # Process 'velocity_6h'
        X['velocity_6h'] = X['velocity_6h'].clip(lower=0)
        
        return X

# 2. Boolean Feature Engineering 
class BooleanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No fitting needed for this transformer

    def transform(self, X):
        X = X.copy()
        X['total_valid_phones'] = X['phone_home_valid'] + X['phone_mobile_valid']
        X['foreign_long_session'] = (X['foreign_request'] & X['keep_alive_session']).astype(int)
        X['device_and_account_history'] = X['device_fraud_count'] * X['keep_alive_session']
        X['total_risk_flags'] = X[['email_is_free', 'foreign_request', 'has_other_cards', 'keep_alive_session']].sum(axis=1)
        return X

# 3. Low Cardinality Feature Engineering 
class LowCardinalityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.onehot_features = ['payment_type', 'employment_status', 'housing_status', 'device_os', 'month', 'source']
        self.ordinal_features = ['customer_age', 'income', 'proposed_credit_limit']
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.ordinal_encoder = OrdinalEncoder()
        
    
    def fit(self, X, y=None):
        # Fit the encoders
        self.onehot_encoder.fit(X[self.onehot_features])
        self.ordinal_encoder.fit(X[self.ordinal_features])
        
        return self
    
    def transform(self, X):
        X = X.copy()
        # One-Hot Encode the specified categorical features
        onehot_encoded = self.onehot_encoder.transform(X[self.onehot_features])
        onehot_feature_names = self.onehot_encoder.get_feature_names_out(self.onehot_features)
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_feature_names, index=X.index)

        # Ordinal Encode the specified ordinal features
        ordinal_encoded = self.ordinal_encoder.transform(X[self.ordinal_features])
        ordinal_df = pd.DataFrame(ordinal_encoded, columns=self.ordinal_features, index=X.index)
        
        # Drop the original one-hot and ordinal features from the DataFrame
        X = X.drop(columns=self.onehot_features + self.ordinal_features)
        
        # Concatenate the encoded features with the remaining columns
        transformed_df = pd.concat([X, onehot_df, ordinal_df], axis=1)
        
        return transformed_df


# 4. High Cardinality Feature Engineering 
class HighCardinalityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = ['prev_address_months_count', 'current_address_months_count', 'bank_months_count']
        self.means_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            # Compute mean of non-missing values (excluding -1)
            valid_values = X[X[col] != -1][col]
            self.means_[col] = valid_values.mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            # Create missing value flag
            missing_flag = X[col] == -1
            X[col + '_missing'] = missing_flag.astype(int)
            # Impute missing values with mean
            X[col] = X[col].replace(-1, self.means_[col])
            if col == 'current_address_months_count':
                # Apply log transformation
                X[col] = np.log1p(X[col])
        return X

# Outlier remover transformer
class OutlierRemoverTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, percentage=0.05):
        self.percentage = percentage
        self.means_ = None
        self.numerical_columns_ = None

    def fit(self, X, y=None):
        X = X.copy()
        # Identify numerical columns
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns
        # Calculate means of numerical features
        self.means_ = X[self.numerical_columns_].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.numerical_columns_:
            mean = self.means_[col]
            # Calculate absolute difference from the mean
            abs_diff = (X[col] - mean).abs()
            # Determine the threshold for the top N% of absolute differences
            threshold = np.percentile(abs_diff, 100 * (1 - self.percentage))
            # Identify outliers (values beyond the threshold)
            outliers = abs_diff >= threshold
            # Impute outliers to the mean
            X.loc[outliers, col] = mean
        return X

# Standard scaling transformer
scaler_transformer = FunctionTransformer(
    lambda X: pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
)
# Combined Preprocessing Pipeline
# ===============================
preprocessor = Pipeline([
    ('numerical_processing', NumericalTransformer()),
    ('boolean_processing', BooleanTransformer()),
    ('low_card_categorical_processing', LowCardinalityTransformer()),
    ('high_card_categorical_processing', HighCardinalityTransformer()),
    (('standard_scaler', scaler_transformer)),
    ('outlier_remover', OutlierRemoverTransformer(percentage=0.05))
])