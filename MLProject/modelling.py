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
import sys

# Flush output immediately
sys.stdout.flush()
sys.stderr.flush()

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
    
    # Disable autolog to prevent interference
    mlflow.sklearn.autolog(disable=True)
    
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
    
    # Disable autolog to prevent interference
    print("\nDisabling autolog...")
    mlflow.sklearn.autolog(disable=True)
    
    # Start MLflow run (WITHOUT autolog to avoid issues)
    print("Starting MLflow run...")
    with mlflow.start_run(run_name="RandomForest_Basic_Autolog"):
        
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")
        
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
        print("✓ Model training completed!")
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = model.predict(X_test)
        print("✓ Predictions completed!")
        
        # Calculate metrics
        print("\nCalculating metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("✓ Metrics calculated!")
        
        # Log parameters
        print("\nLogging parameters...")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("random_state", 42)
        print("✓ Parameters logged!")
        
        # Log metrics
        print("\nLogging metrics...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print("✓ Metrics logged!")
        
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
        
        # LOG MODEL - This is critical!
        print("\n" + "="*60)
        print("LOGGING MODEL TO MLFLOW")
        print("="*60)
        sys.stdout.flush()
        
        try:
            print(f"Model object type: {type(model)}")
            print(f"Model parameters: {model.get_params()}")
            sys.stdout.flush()
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )
            print("✓✓✓ Model logged successfully! ✓✓✓")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"❌ Error logging model: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            
            print("\n" + "="*60)
            print("TRYING ALTERNATIVE METHOD")
            print("="*60)
            sys.stdout.flush()
            
            import joblib
            import tempfile
            import os
            
            # Save model manually if log_model fails
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pkl")
                print(f"Saving model to: {model_path}")
                joblib.dump(model, model_path)
                print(f"Model saved, file size: {os.path.getsize(model_path)} bytes")
                
                mlflow.log_artifact(model_path, "model")
                print("✓ Model saved via alternative method!")
            sys.stdout.flush()
        
        # Verify model is saved
        import os
        artifact_uri = mlflow.get_artifact_uri()
        print("\n" + "="*60)
        print("VERIFYING MODEL ARTIFACTS")
        print("="*60)
        print(f"Artifact URI: {artifact_uri}")
        
        # Check if artifacts directory exists
        artifacts_path = artifact_uri.replace("file://", "")
        if os.path.exists(artifacts_path):
            print(f"✓ Artifacts directory exists!")
            print(f"Contents:")
            for root, dirs, files in os.walk(artifacts_path):
                level = root.replace(artifacts_path, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f'{subindent}{file}')
        else:
            print(f"❌ Artifacts directory NOT found!")
        
        sys.stdout.flush()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"✓ MLflow artifacts saved to: {artifact_uri}")
        print(f"✓ Run ID: {mlflow.active_run().info.run_id}")
        sys.stdout.flush()

if __name__ == "__main__":
    train_model_with_autolog()
