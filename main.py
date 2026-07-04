"""
AIMS Kaggle 2K26 - Rank 46 Submission
Ensemble Model: Advanced Bagging, Boosting & Stacking

This script preprocesses the planetary dataset, engineers domain-specific features, 
and trains a multi-level stacking/voting ensemble to predict habitability classifications.
"""

import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# FEATURE ENGINEERING

def create_features(df):
    """Generates domain-specific interaction features based on planetary physics."""
    df = df.copy()
    
    if 'Surface Temperature' in df.columns and 'Water Content' in df.columns:
        df['Habitability_Index'] = df['Surface Temperature'] * df['Water Content']
        df['Earth_Like_Zone'] = ((df['Surface Temperature'] > -2) & 
                                 (df['Surface Temperature'] < 2) & 
                                 (df['Water Content'] > -1)).astype(int)
    
    if 'Gravity' in df.columns and 'Atmospheric Density' in df.columns:
        df['Atmosphere_Retention'] = df['Gravity'] * df['Atmospheric Density']
    
    if 'Proximity to Star' in df.columns and 'Stellar Luminosity' in df.columns:
        df['Radiation_Exposure'] = df['Stellar Luminosity'] / (df['Proximity to Star']**2 + 0.1)
    
    if 'Orbital Period' in df.columns and 'Proximity to Star' in df.columns:
        df['Kepler_Ratio'] = (df['Orbital Period']**2) / (df['Proximity to Star']**3 + 0.1)
    
    if 'Surface Temperature' in df.columns and 'Surface Pressure' in df.columns:
        df['Surface_Conditions'] = df['Surface Temperature'] * df['Surface Pressure']
        df['Ice_World_Indicator'] = ((df['Surface Temperature'] < -1) & 
                                     (df['Surface Pressure'] > 0)).astype(int)
    
    if 'Gravity' in df.columns and 'Atmospheric Density' in df.columns and 'Moon Count' in df.columns:
        df['Gas_Giant_Score'] = (df['Gravity'] > 0.5).astype(int) + \
                                (df['Atmospheric Density'] > 0.5).astype(int) + \
                                (df['Moon Count'] >= 3).astype(int)
    
    if 'Mineral Abundance' in df.columns and 'Gravity' in df.columns:
        df['Rocky_Planet_Score'] = df['Mineral Abundance'] * df['Gravity']
    
    if 'Surface Temperature' in df.columns and 'Gravity' in df.columns:
        df['Atmosphere_Escape_Risk'] = df['Surface Temperature'] / (df['Gravity'] + 1)
    
    if 'Stellar Luminosity' in df.columns and 'Proximity to Star' in df.columns:
        df['Energy_Received'] = df['Stellar Luminosity'] / (df['Proximity to Star'] + 0.1)
    
    if 'Gravity' in df.columns and 'Moon Count' in df.columns:
        df['Mass_Proxy'] = df['Gravity'] * (df['Moon Count'] + 1)
        
    return df


def create_separator_features(df, X_train_ref, y_train_ref):
    """Creates weighted separator features to differentiate highly overlapping classes."""
    df = df.copy()
    common_features = list(set(X_train_ref.columns) & set(df.columns))

    # Helper function to generate separators based on class mean differences
    def generate_separator(class_a, class_b, weights):
        mean_a = X_train_ref[y_train_ref == class_a][common_features].mean()
        mean_b = X_train_ref[y_train_ref == class_b][common_features].mean()
        diff = abs(mean_a - mean_b).sort_values(ascending=False)
        top_features = [f for f in diff.head(len(weights)).index if f in df.columns]
        
        if len(top_features) == len(weights):
            return sum(df[feat] * weight for feat, weight in zip(top_features, weights))
        return 0

    df['Sep_Class_4_9'] = generate_separator(4, 9, [0.8, 0.1, 0.1])
    df['Sep_Class_3_8'] = generate_separator(3, 8, [0.8, 0.1, 0.1])
    df['Sep_Class_0_5'] = generate_separator(0, 5, [0.6, 0.2, 0.2])
    
    return df

# MAIN EXECUTION PIPELINE

if __name__ == "__main__":
    
    # 1. Load Data (Assumes CSVs are in the same directory)
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    test_ids = test_df['id'].copy()
    train_df = train_df.drop(['id', 'Probe ID'], axis=1, errors='ignore')
    test_df = test_df.drop(['id', 'Probe ID'], axis=1, errors='ignore')

    y_train = train_df['Prediction'].values
    train_df = train_df.drop('Prediction', axis=1)

    # 2. Preprocessing & Imputation
    print("Executing imputation and encoding pipeline...")
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
        test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

    if train_df[num_cols].isnull().sum().sum() > 0 or test_df[num_cols].isnull().sum().sum() > 0:
        knn_imputer = KNNImputer(n_neighbors=7, weights='distance')
        train_df[num_cols] = knn_imputer.fit_transform(train_df[num_cols])
        test_df[num_cols] = knn_imputer.transform(test_df[num_cols])

    if cat_cols:
        train_df['_source'] = 'train'
        test_df['_source'] = 'test'
        combined = pd.concat([train_df, test_df], axis=0)
        combined = pd.get_dummies(combined, columns=cat_cols, dtype=int, drop_first=False)
        train_df = combined[combined['_source'] == 'train'].drop('_source', axis=1).reset_index(drop=True)
        test_df = combined[combined['_source'] == 'test'].drop('_source', axis=1).reset_index(drop=True)

    # 3. Feature Generation
    print("Generating physics-based and synthetic separator features...")
    X_train = create_features(train_df)
    X_test = create_features(test_df)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    X_train = create_separator_features(X_train, X_train, y_train)
    X_test = create_separator_features(X_test, X_train, y_train)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 4. Validation Split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    # 5. Model Definitions
    print("Initializing ensemble base models...")
    
    # Base Boosters
    catboost = CatBoostClassifier(iterations=300, depth=8, random_state=42, verbose=False)
    lightgbm = LGBMClassifier(n_estimators=300, max_depth=8, random_state=42, verbose=-1)
    xgboost = XGBClassifier(n_estimators=300, max_depth=8, random_state=42, verbosity=0)
    gradboost = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)

    # Level 2 Stacking Classifier
    stacking_model = StackingClassifier(
        estimators=[
            ('catboost', catboost),
            ('lightgbm', lightgbm),
            ('xgboost', xgboost),
            ('rf', rf),
            ('gradboost', gradboost)
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        n_jobs=-1
    )

    # 6. Training & Validation
    print("Training Stacking Ensemble (this may take a while)...")
    stacking_model.fit(X_tr, y_tr)
    
    stacking_pred = stacking_model.predict(X_val)
    val_acc = accuracy_score(y_val, stacking_pred)
    
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_val, stacking_pred))

    # 7. Final Training & Prediction
    print("Retraining on full dataset for final submission...")
    stacking_model.fit(X_train, y_train)
    predictions = stacking_model.predict(X_test)

    # 8. Export Submission
    submission = pd.DataFrame({
        'id': test_ids,
        'Prediction': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv successfully.")