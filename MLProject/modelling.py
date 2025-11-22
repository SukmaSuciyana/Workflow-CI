"""
Basic Model Training with MLflow Autolog
Kriteria Basic: Melatih model machine learning menggunakan MLflow dengan autolog
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from pathlib import Path

def load_data(data_path):
    """Load preprocessed training and testing data"""
    X_train = pd.read_csv(data_path / 'X_train.csv')
    X_test = pd.read_csv(data_path / 'X_test.csv')
    y_train = pd.read_csv(data_path / 'y_train.csv')
    y_test = pd.read_csv(data_path / 'y_test.csv')
    
    # Convert to numpy arrays
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    return X_train, X_test, y_train, y_test


def train_model_with_autolog():
    """Train model using MLflow autolog"""
    
    # Set MLflow tracking URI to local directory
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set experiment name
    mlflow.set_experiment("pipe_condition_classification_basic")
    
    # Load preprocessed data
    data_path = Path(__file__).parent / 'preprocessed_data_auto'
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    print("Data loaded successfully!")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Classes: {np.unique(y_train)}")
    
    # Enable autolog for sklearn
    mlflow.sklearn.autolog()
    
    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest_Basic_Autolog"):
        
        # Train model
        print("\nTraining Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics (autolog will also log these)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print("\n=== Model Performance ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Good', 'Moderate', 'Critical']))
        
        print("\n=== Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))
        
        print("\n✓ Model training completed!")
        print(f"✓ MLflow artifacts saved to: {mlflow.get_artifact_uri()}")
        print(f"✓ Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train_model_with_autolog()
