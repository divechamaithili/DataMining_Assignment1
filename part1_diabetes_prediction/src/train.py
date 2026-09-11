import os, json, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from data_preprocessing import prepare_data

def train():
    os.makedirs('../results', exist_ok=True)
    X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names, X_test = prepare_data()
    model = HistGradientBoostingClassifier(
        class_weight='balanced', learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=0.5, max_iter=150, random_state=42
    )
    model.fit(X_train_proc, y_train)
    joblib.dump(model, '../results/best_model.joblib')
    joblib.dump(preprocessor, '../results/preprocessor.joblib')
    print("Model training complete. Artifacts saved in results/")

if __name__ == '__main__':
    train()
