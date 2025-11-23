"""
Basic Model Training with MLflow
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from pathlib import Path
import sys
import os

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


def train_model():
    """Train model and log with MLflow"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set experiment
    mlflow.set_experiment("pipe_condition_classification_basic")
    
    # Load data - check multiple possible locations
    script_dir = Path(__file__).parent
    
    # Try MLProject/preprocessed_data_auto first (local development)
    data_path = script_dir / 'preprocessed_data_auto'
    if not data_path.exists():
        # Try parent directory (GitHub Actions)
        data_path = script_dir.parent / 'preprocessed_data_auto'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found at {data_path}")
    
    print(f"Loading data from: {data_path}")
    sys.stdout.flush()
    
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    print("Data loaded successfully!")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Classes: {np.unique(y_train)}")
    sys.stdout.flush()
    
    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest_Model"):
        
        print(f"\nRun ID: {mlflow.active_run().info.run_id}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")
        sys.stdout.flush()
        
        # Train model
        print("\nTraining Random Forest model...")
        sys.stdout.flush()
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        print("✓ Model training completed!")
        sys.stdout.flush()
        
        # Predictions
        print("\nMaking predictions...")
        y_pred = model.predict(X_test)
        print("✓ Predictions completed!")
        sys.stdout.flush()
        
        # Calculate metrics
        print("\nCalculating metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("✓ Metrics calculated!")
        sys.stdout.flush()
        
        # Log parameters
        print("\nLogging parameters...")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("random_state", 42)
        print("✓ Parameters logged!")
        sys.stdout.flush()
        
        # Log metrics
        print("\nLogging metrics...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print("✓ Metrics logged!")
        sys.stdout.flush()
        
        # Print performance
        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Good', 'Moderate', 'Critical']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        sys.stdout.flush()
        
        # LOG MODEL - CRITICAL STEP
        print("\n" + "="*60)
        print("LOGGING MODEL")
        print("="*60)
        sys.stdout.flush()
        
        # Method: Use mlflow.sklearn.log_model with proper parameters
        print("\nLogging model with mlflow.sklearn.log_model...")
        sys.stdout.flush()
        
        try:
            # Create input example from first row of test data
            input_example = X_test.iloc[:1]
            
            # Log model - use 'artifact_path' for compatibility
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example
            )
            print(f"✓ Model logged to artifact_path: model")
            print(f"✓ Model info: {model_info}")
            sys.stdout.flush()
            
            # Force flush to disk - sometimes MLflow doesn't write immediately
            import time
            time.sleep(2)
            
            # Verify the model was actually saved
            artifact_uri = mlflow.get_artifact_uri()
            artifact_path = artifact_uri.replace("file://", "")
            model_dir = os.path.join(artifact_path, "model")
            
            print(f"\nVerifying model directory: {model_dir}")
            
            # Check if directory exists
            if not os.path.exists(model_dir):
                print(f"❌ Model directory not found: {model_dir}")
                print(f"Checking parent directory: {artifact_path}")
                if os.path.exists(artifact_path):
                    print("Parent directory contents:")
                    for item in os.listdir(artifact_path):
                        print(f"  - {item}")
                raise Exception(f"Model directory not created at {model_dir}")
            
            print(f"✓ Model directory exists!")
            print("Directory contents:")
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                size = os.path.getsize(item_path) if os.path.isfile(item_path) else 0
                print(f"  - {item} ({size} bytes)")
            
            # Check for MLmodel file specifically
            mlmodel_file = os.path.join(model_dir, "MLmodel")
            if not os.path.exists(mlmodel_file):
                print(f"❌ MLmodel file not found at {mlmodel_file}")
                raise Exception(f"MLmodel file NOT created at {mlmodel_file}")
            
            print(f"✓ VERIFIED: MLmodel file exists!")
            print("\n✓✓✓ Model logged and verified successfully! ✓✓✓")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"❌ Model logging failed: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise  # Re-raise to fail the workflow
        
        # Verify artifacts
        print("\n" + "="*60)
        print("VERIFYING ARTIFACTS")
        print("="*60)
        
        artifact_uri = mlflow.get_artifact_uri()
        artifact_path = artifact_uri.replace("file://", "")
        
        print(f"Artifact URI: {artifact_uri}")
        print(f"Artifact Path: {artifact_path}")
        
        if os.path.exists(artifact_path):
            print("\n✓ Artifacts directory exists!")
            print("\nDirectory contents:")
            for root, dirs, files in os.walk(artifact_path):
                level = root.replace(artifact_path, '').count(os.sep)
                indent = '  ' * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = '  ' * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f'{subindent}{file} ({file_size} bytes)')
        else:
            print(f"\n❌ Artifacts directory NOT found at: {artifact_path}")
        
        sys.stdout.flush()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"✓ Run ID: {mlflow.active_run().info.run_id}")
        print(f"✓ Artifacts saved to: {artifact_uri}")
        print("="*60)
        sys.stdout.flush()

if __name__ == "__main__":
    train_model()
