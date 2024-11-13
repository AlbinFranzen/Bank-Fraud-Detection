# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, mutual_info_classif, RFE
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# Helper Functions
# =================

def plot_feature_importances(df, x, y, title, xlabel='Features', ylabel='Importance Value'):
    plt.figure(figsize=(16, 8))
    sns.barplot(data=df.sort_values(by=y, ascending=False), x=x, y=y, palette='viridis', alpha=.6)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# Preprocessing Pipelines
# =======================

# Numerical Preprocessor
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', add_indicator=True)),
    ('scaler', RobustScaler())
])

# Low Card Preprocessor
low_card_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# High Card Preprocessor
high_card_pipeline = Pipeline([
    ('ordinal', OrdinalEncoder())
])

# Boolean Feature Engineering Pipeline
boolean_pipeline = Pipeline([
    ('total_valid_phones', FunctionTransformer(
        lambda X: pd.DataFrame((X['phone_home_valid'] + X['phone_mobile_valid']).values.reshape(-1, 1), columns=['total_valid_phones'])
    )),
    ('foreign_long_session', FunctionTransformer(
        lambda X: pd.DataFrame((X['foreign_request'] & X['keep_alive_session']).astype(int).values.reshape(-1, 1), columns=['foreign_long_session'])
    )),
    ('device_and_account_history', FunctionTransformer(
        lambda X: pd.DataFrame((X['device_fraud_count'] * X['keep_alive_session']).values.reshape(-1, 1), columns=['device_and_account_history'])
    )),
    ('total_risk_flags', FunctionTransformer(
        lambda X: pd.DataFrame(X[['email_is_free', 'foreign_request', 'has_other_cards', 'keep_alive_session']].sum(axis=1).values.reshape(-1, 1), columns=['total_risk_flags'])
    ))
])

# Scaling and Selection Pipeline
scaling_and_selection_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('variance_threshold', VarianceThreshold())
])

# Combined Preprocessing Pipeline
# ===============================
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('numerical', numerical_pipeline, selector(dtype_exclude='object')),
#         ('categorical', categorical_pipeline, selector(dtype_include='object')),
#         ('boolean', boolean_pipeline, ['phone_home_valid', 'phone_mobile_valid', 'foreign_request', 'keep_alive_session'])
#     ],
#     remainder='drop'
# )

# Feature Selection and Evaluation
# ================================
def feature_selection(X_scaled_nm, y_scaled_nm):
    # Extra Trees Classifier for Feature Importance
    extra = ExtraTreesClassifier(n_estimators=50, random_state=0)
    extra.fit(X_scaled_nm, y_scaled_nm)
    feature_sel_extra = SelectFromModel(extra, prefit=True)
    best_extra_features = X_scaled_nm.columns[feature_sel_extra.get_support()].tolist()
    print(best_extra_features)

    # Plot Feature Importances
    extra_importances = pd.DataFrame({'feature': X_scaled_nm.columns, 'importance': extra.feature_importances_, 'model': 'ExtraTreesClassifier'})
    plot_feature_importances(extra_importances, 'feature', 'importance', 'Feature Importances by Model')

    # Mutual Information Test for Numeric Features with Hyperparameter Optimization
    numeric_features = [feature for feature in X_scaled_nm.columns if X_scaled_nm[feature].nunique() >= 10]
    X_train_num = X_scaled_nm[numeric_features]
    param_grid = {'k': range(5, min(20, len(numeric_features)))}
    grid_search = GridSearchCV(SelectKBest(mutual_info_classif), param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train_num, y_scaled_nm)
    best_k = grid_search.best_params_['k']
    best_mutual_info_cols = SelectKBest(mutual_info_classif, k=best_k)
    best_mutual_info_cols.fit(X_train_num, y_scaled_nm)
    best_mutual_info_features = X_train_num.columns[best_mutual_info_cols.get_support()].tolist()
    print("Best features selected by Mutual Information with k optimization:", best_mutual_info_features)

    # Recursive Feature Elimination (RFE)
    rf = RandomForestClassifier(n_estimators=50, random_state=0)
    rfe = RFE(estimator=rf, n_features_to_select=10)
    rfe.fit(X_scaled_nm, y_scaled_nm)
    best_rfe_features = X_scaled_nm.columns[rfe.get_support()].tolist()
    print("Best features selected by RFE:", best_rfe_features)

# Resampling Pipeline
# ===================
def resampling_pipeline(X, y):
    print(f'Test dataset samples per class {Counter(y)}')
    
    # SMOTE for Over-Sampling
    smote = SMOTE(sampling_strategy='minority', random_state=0)
    X_resampled_sm, y_resampled_sm = smote.fit_resample(X, y)
    print('After SMOTE resampling dataset shape %s' % Counter(y_resampled_sm))
    
    # NearMiss for Under-Sampling
    nm = NearMiss(sampling_strategy=1, n_jobs=-1)
    X_resampled_nm, y_resampled_nm = nm.fit_resample(X_resampled_sm, y_resampled_sm)
    print('After NearMiss resampling dataset shape %s' % Counter(y_resampled_nm))
    
    return X_resampled_nm, y_resampled_nm

# Final Usage Example
# ===================
# transformed_data = preprocessor.fit_transform(train_df)
# You can now use the transformed data for training/testing models or further analysis.

# Variance Threshold Test for Constant Features
# ==============================================
def variance_threshold_test(X_scaled_nm):
    selector = VarianceThreshold()
    selector.fit(X_scaled_nm)
    constant_features = [feature for feature in X_scaled_nm.columns if feature not in X_scaled_nm.columns[selector.get_support()]]
    print(constant_features)
    X_scaled_nm.drop(constant_features, axis=1, inplace=True)





