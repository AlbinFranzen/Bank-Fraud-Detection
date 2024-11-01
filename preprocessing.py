import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd

# Column-specific transformers for imputing -1 as missing values and applying specific imputations
numerical_preprocessor = ColumnTransformer(
    transformers=[
        # Replace -1 with NaN and impute `session_length_in_minutes` with the median
        ('session_length_median_imputer', 
         Pipeline([
             ('replace_minus_one', SimpleImputer(missing_values=-1, strategy='constant', fill_value=np.nan)),
             ('median_imputer', SimpleImputer(strategy='median'))
         ]), 
         ['session_length_in_minutes']),
        
        # Replace -1 with NaN and impute `intended_balcon_amount` with 0
        ('intended_balcon_zero_imputer', 
         Pipeline([
             ('replace_minus_one', SimpleImputer(missing_values=-1, strategy='constant', fill_value=np.nan)),
             ('zero_imputer', SimpleImputer(strategy='constant', fill_value=0))
         ]), 
         ['intended_balcon_amount'])
    ],
    remainder='passthrough'
)

# Usage example:
# data = pd.DataFrame(...)  # Your data here
# transformed_data = numerical_preprocessor.fit_transform(data)


