import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath='diabetes_prediction_dataset.csv'):
    if not os.path.exists(filepath):
        for alt in ['..', '../..', 'data', '../data']:
            p = os.path.join(alt, filepath)
            if os.path.exists(p):
                return pd.read_csv(p)
        raise FileNotFoundError(f"Could not locate {filepath}. Please place it in the project root or data/ directory.")
    return pd.read_csv(filepath)

def engineer_features(df):
    df_feat = df.copy()
    smoking_map = {
        'never': 'never', 'current': 'current', 'former': 'former_or_ever',
        'not current': 'former_or_ever', 'ever': 'former_or_ever', 'No Info': 'unknown'
    }
    df_feat['smoking_history'] = df_feat['smoking_history'].map(smoking_map)
    df_feat['bmi_was_imputed'] = (np.abs(df_feat['bmi'] - 27.32) < 0.001).astype(int)
    df_feat['glucose_x_hba1c'] = df_feat['blood_glucose_level'] * df_feat['HbA1c_level']
    df_feat['comorbidity_count'] = df_feat['hypertension'] + df_feat['heart_disease']
    return df_feat

def prepare_data(filepath='diabetes_prediction_dataset.csv', test_size=0.20, random_state=42):
    raw_df = load_data(filepath)
    df_feat = engineer_features(raw_df)
    X = df_feat.drop(columns=['diabetes'])
    y = df_feat['diabetes'].astype(int)

    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_combined = pd.concat([X_train_raw, y_train_raw], axis=1).drop_duplicates()
    X_train = train_combined.drop(columns=['diabetes'])
    y_train = train_combined['diabetes']

    num_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'glucose_x_hba1c']
    cat_features = ['gender', 'smoking_history']
    passthrough_features = ['hypertension', 'heart_disease', 'bmi_was_imputed', 'comorbidity_count']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_features),
            ('pass', 'passthrough', passthrough_features)
        ]
    )
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return preprocessor.transform(X_train), preprocessor.transform(X_test), y_train, y_test, preprocessor, feature_names, X_test
