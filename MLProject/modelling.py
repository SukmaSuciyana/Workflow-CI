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
    mlflow.set_tracking_uri("file:./MLProject/mlruns")

    # Set experiment
    mlflow.set_experiment("pipe_condition_classification_basic")
    
    # Load data
    data_path = Path(__file__).parent / 'preprocessed_data_auto'
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
        
        try:
            # Method 1: Use mlflow.sklearn.log_model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                conda_env=None,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            )
            print("✓✓✓ Model logged successfully with mlflow.sklearn.log_model! ✓✓✓")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"❌ Method 1 failed: {e}")
            print("\nTrying Method 2: Direct pickle save...")
            sys.stdout.flush()
            
            try:
                import pickle
                import tempfile
                
                # Create temp directory for model
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_file = os.path.join(tmpdir, "model.pkl")
                    
                    # Save model with pickle
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                    
                    print(f"Model saved to {model_file}")
                    print(f"File size: {os.path.getsize(model_file)} bytes")
                    
                    # Log as artifact
                    mlflow.log_artifact(model_file, artifact_path="model")
                    
                print("✓✓✓ Model saved with Method 2! ✓✓✓")
                sys.stdout.flush()
                
            except Exception as e2:
                print(f"❌ Method 2 also failed: {e2}")
                print("\nERROR: Could not save model!")
                sys.stdout.flush()
                raise
        
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
