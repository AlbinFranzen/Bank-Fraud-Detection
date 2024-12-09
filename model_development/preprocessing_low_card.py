import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class LowCardinalityPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
       
        # one-hot encoded features
        self.onehot_features = [
            'payment_type',
            'employment_status',
            'housing_status',
            'device_os',
            'month',
            'credit_bin', # proposed_credit_limit
            'income_bin' # income
        ]
        self.onehot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
        
        # ordinal encoded feature bc shows linear relationship
        self.ordinal_features = ['customer_age']
        self.ordinal_encoder = OrdinalEncoder()
        
        
    def create_bins(self, X):
        X_transformed = X.copy()
        
        # bin credit limit (U-shaped relationship)
        credit_bins = [0, 500, 1000, 2000, 5000, 10000, float('inf')]
        X_transformed['credit_bin'] = pd.cut(X['proposed_credit_limit'], 
                                           bins=credit_bins, 
                                           labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme'])
        
        # bin income (U-shaped relationship)
        income_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        X_transformed['income_bin'] = pd.cut(X['income'], 
                                           bins=income_bins, 
                                           labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        return X_transformed
        
    def fit(self, X, y=None):
        
        # creating bins and fitting categorical features
        X_binned = self.create_bins(X)
        self.onehot_encoder.fit(X_binned[self.onehot_features])
        self.ordinal_encoder.fit(X[self.ordinal_features])
        
        return self
    
    def transform(self, X):
        # create bins
        X_transformed = self.create_bins(X)
        
        # transform form one-hot features --> to dense array
        onehot_encoded = self.onehot_encoder.transform(X_transformed[self.onehot_features]).toarray()
        onehot_feature_names = self.onehot_encoder.get_feature_names_out(self.onehot_features)
        
        # transform ordinal feature (customer_age)
        ordinal_encoded = self.ordinal_encoder.transform(X[self.ordinal_features])
        
        # combine all transformed features
        transformed_array = np.hstack([onehot_encoded, ordinal_encoded])
        
        # create feature names
        feature_names = [*onehot_feature_names, *self.ordinal_features]
        
        # create final dataframe
        transformed_df = pd.DataFrame(
            transformed_array,
            columns=feature_names,
            index=X.index
        )
        
        # separately handle device_distinct_emails_8w since it's already numeric
        transformed_df['device_distinct_emails_8w'] = X['device_distinct_emails_8w']
        
        return transformed_df

