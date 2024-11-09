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

# Custom Transformers and Helper Functions
# =======================================

# 1. Custom Transformer: Outlier Removal
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3.0):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[np.abs(X) < self.threshold]

# 2. Custom Transformer: Missing Flagger
class MissingFlagger(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_flag=None):
        self.columns_to_flag = columns_to_flag
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns_to_flag:
            X_transformed[f'MISSING_FLAG_{col}'] = X_transformed[col].isnull().astype(int)
        return X_transformed

# 3. Custom Transformer: Categorical Converter
class CategoricalConverter(BaseEstimator, TransformerMixin):
    def __init__(self, cat_columns):
        self.cat_columns = cat_columns
        self.categories_ = {}

    def fit(self, X, y=None):
        for col in self.cat_columns:
            self.categories_[col] = X[col].astype('category').cat.categories
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cat_columns:
            X_transformed[col] = pd.Categorical(X_transformed[col],
                                                categories=self.categories_[col], 
                                                ordered=False)
        return X_transformed    

# 4. Custom Transformer: One-Hot Encoder
class CustomOneHotEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, ohe_columns):
        self.ohe_columns = ohe_columns
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        self.ohe.fit(X[self.ohe_columns].astype('category'))
        return self

    def transform(self, X):
        # One-hot encode the specified columns
        X_ohe = self.ohe.transform(X[self.ohe_columns])
        ohe_column_names = self.ohe.get_feature_names_out(self.ohe_columns)
        X_ohe = pd.DataFrame(X_ohe, columns=ohe_column_names, index=X.index)

        # Concatenate the one-hot-encoded columns with the remaining columns
        X_transformed = pd.concat([X.drop(self.ohe_columns, axis=1), X_ohe], axis=1)
        return X_transformed

# Preprocessing Pipelines
# =======================

# 5. Numerical Preprocessor
def numerical_pipeline():
    return Pipeline([
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('name_email_similarity', Pipeline([
                    ('binning', FunctionTransformer(lambda X: pd.cut(X.squeeze(), bins=[0, 0.24, 0.48, 0.72, 0.96, 1.0], labels=False).values.reshape(-1, 1), validate=False)),
                    ('ordinal_encoder', OrdinalEncoder()),
                ]), ['name_email_similarity']),

                ('days_since_request', Pipeline([
                    ('log_transform', FunctionTransformer(lambda X: pd.DataFrame(np.log1p(X.to_numpy())), validate=False)),
                ]), ['days_since_request']),

                ('intended_balcon_amount', Pipeline([
                    ('missing_flag', FunctionTransformer(lambda x: (x == -1).astype(int), validate=False)),
                    ('imputer', SimpleImputer(missing_values=-1, strategy='constant', fill_value=0)),
                    ('log_transform', FunctionTransformer(np.log1p, validate=False)),
                ]), ['intended_balcon_amount']),

                ('velocity_6h', Pipeline([
                    ('negative_imputer', SimpleImputer(missing_values=-1, strategy='constant', fill_value=0)),
                ]), ['velocity_6h'])
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )),
        ('drop_columns', FunctionTransformer(lambda X: X.drop(columns=['velocity_4w', 'session_length_in_minutes']) if isinstance(X, pd.DataFrame) else X, validate=False))
    ])

# 6. Boolean Feature Engineering Pipeline
def boolean_pipeline():
    return ColumnTransformer(
        transformers=[
            ('total_valid_phones', FunctionTransformer(lambda X: (X['phone_home_valid'] + X['phone_mobile_valid']).values.reshape(-1, 1)), ['phone_home_valid', 'phone_mobile_valid']),
            ('foreign_long_session', FunctionTransformer(lambda X: (X['foreign_request'] & X['keep_alive_session']).astype(int).values.reshape(-1, 1)), ['foreign_request', 'keep_alive_session']),
            ('device_and_account_history', FunctionTransformer(lambda X: (X['device_fraud_count'] * X['keep_alive_session']).values.reshape(-1, 1)), ['device_fraud_count', 'keep_alive_session']),
            ('total_risk_flags', FunctionTransformer(lambda X: X[['email_is_free', 'foreign_request', 'has_other_cards', 'keep_alive_session']].sum(axis=1).values.reshape(-1, 1)), ['email_is_free', 'foreign_request', 'has_other_cards', 'keep_alive_session']),
            ('source_encoded', 'passthrough', ['source'])
        ],
        remainder='passthrough'
    )

# 7. Low Cardinality Feature Engineering Pipeline
def low_cardinality_pipeline():
    return Pipeline([
        # Create bins for 'proposed_credit_limit' and 'income', then drop specified columns
        ('preprocess', FunctionTransformer(
            lambda X: (
                X.assign(
                    credit_bin=pd.cut(X['proposed_credit_limit'], bins=[0, 500, 1000, 2000, 5000, 10000, float('inf')], 
                                      labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme']),
                    income_bin=pd.cut(X['income'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                      labels=['very_low', 'low', 'medium', 'high', 'very_high'])
                )
                .drop(columns=['velocity_4w', 'session_length_in_minutes'], errors='ignore')
            ), validate=False
        )),
        
        # Encode categorical features
        ('encode', ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), [
                    'payment_type', 'employment_status', 'housing_status', 
                    'device_os', 'month', 'credit_bin', 'income_bin'
                ]),
                ('ordinal', OrdinalEncoder(), ['customer_age'])
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ))
    ])

# 8. High Cardinality Feature Engineering Pipeline
def high_cardinality_pipeline():
    high_card_preprocessor = ColumnTransformer([
        ('ordinal_encoding', CategoricalConverter(['high_card']), 'high_card')
    ], remainder='passthrough')

    return high_card_preprocessor

# 9. Scaling and Selection Pipeline
def scaling_and_selection_pipeline():
    min_max_transformer = FunctionTransformer(lambda X: pd.concat([X.drop(columns=X.select_dtypes(include=['float64', 'int64']).columns), pd.DataFrame(MinMaxScaler().fit_transform(X.select_dtypes(include=['float64', 'int64'])), columns=X.select_dtypes(include=['float64', 'int64']).columns, index=X.index)], axis=1), validate=False)
    variance_selector = VarianceThreshold()
    return min_max_transformer, variance_selector

# 10. Resampling Pipeline
def resampling_pipeline(X, y):
    print(f'Test dataset samples per class {Counter(y)}')
    nm = NearMiss(sampling_strategy=1, n_jobs=-1)
    X_scaled_nm, y_scaled_nm = nm.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_scaled_nm))
    return X_scaled_nm, y_scaled_nm

# Combined Preprocessing Pipeline
# ===============================
numerical_preprocessor = numerical_pipeline()
boolean_preprocessor = boolean_pipeline()
low_card_preprocessor = low_cardinality_pipeline()
high_card_preprocessor = high_cardinality_pipeline()
min_max_transformer, variance_selector = scaling_and_selection_pipeline()

preprocessor = Pipeline([
    ('numerical_processing', numerical_preprocessor),
    ('boolean_processing', boolean_preprocessor),
    ('low_card_categorical_processing', low_card_preprocessor),
    ('high_card_categorical_processing', high_card_preprocessor),
    ('min_max_scaling', min_max_transformer),
    ('variance_threshold', variance_selector)
])

# Feature Selection and Evaluation
# ================================
def feature_selection(X_scaled_nm, y_scaled_nm):
    extra = ExtraTreesClassifier(n_estimators=50, random_state=0)
    extra.fit(X_scaled_nm, y_scaled_nm)
    feature_sel_extra = SelectFromModel(extra, prefit=True)
    best_extra_features = X_scaled_nm.columns[feature_sel_extra.get_support()].tolist()
    print(best_extra_features)

    # Plot Feature Importances
    extra_importances = pd.DataFrame({'feature': X_scaled_nm.columns, 'importance': extra.feature_importances_, 'model': 'ExtraTreesClassifier'})
    plt.figure(figsize=(16, 8))
    sns.barplot(data=extra_importances.sort_values(by='importance', ascending=False), x="feature", y="importance", hue="model", palette='viridis', alpha=.6)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Importance Value", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Feature Importances by Model", fontsize=14)
    plt.legend(title='Model', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Mutual Information Test for Numeric Features
    numeric_features = [feature for feature in X_scaled_nm.columns if X_scaled_nm[feature].nunique() >= 10]
    X_train_num = X_scaled_nm[numeric_features]
    mutual_info_results = mutual_info_classif(X_train_num, y_scaled_nm)
    mutual_info_results_df = pd.DataFrame({'feature': X_train_num.columns, 'mutual_info': mutual_info_results})

    plt.figure(figsize=(16, 8))
    sns.barplot(data=mutual_info_results_df.sort_values(by='mutual_info', ascending=False), x="feature", y="mutual_info", palette='viridis', alpha=.6)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Mutual Information Value", fontsize=12)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Mutual Information Value by Numerical Feature", fontsize=14)
    plt.tight_layout()
    plt.show()

    best_mutual_info_cols = SelectKBest(mutual_info_classif, k=15)
    best_mutual_info_cols.fit(X_train_num, y_scaled_nm)
    best_mutual_info_features = X_train_num.columns[best_mutual_info_cols.get_support()].tolist()
    print(best_mutual_info_features)

# Variance Threshold Test for Constant Features
# ==============================================
def variance_threshold_test(X_scaled_nm):
    selector = VarianceThreshold()
    selector.fit(X_scaled_nm)
    constant_features = [feature for feature in X_scaled_nm.columns if feature not in X_scaled_nm.columns[selector.get_support()]]
    print(constant_features)
    X_scaled_nm.drop(constant_features, axis=1, inplace=True)

# Final Usage Example
# ===================
# transformed_data = preprocessor.fit_transform(train_df)
# You can now use the transformed data for training/testing models or further analysis.


