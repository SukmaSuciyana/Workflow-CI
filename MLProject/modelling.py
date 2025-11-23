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
        print("LOGGING MODEL - MANUAL APPROACH")
        print("="*60)
        sys.stdout.flush()
        
        try:
            import pickle
            import yaml
            
            # Get artifact path
            artifact_uri = mlflow.get_artifact_uri()
            artifact_path = artifact_uri.replace("file://", "")
            model_dir = os.path.join(artifact_path, "model")
            
            print(f"Creating model directory: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
            sys.stdout.flush()
            
            # 1. Save model with pickle
            model_pkl_path = os.path.join(model_dir, "model.pkl")
            print(f"Saving model to: {model_pkl_path}")
            with open(model_pkl_path, 'wb') as f:
                pickle.dump(model, f)
            
            file_size = os.path.getsize(model_pkl_path)
            print(f"✓ Model saved: {file_size} bytes")
            sys.stdout.flush()
            
            # 2. Create conda.yaml
            conda_env = {
                'name': 'mlflow-env',
                'channels': ['conda-forge'],
                'dependencies': [
                    'python=3.9',
                    'pip',
                    {'pip': [
                        'mlflow==2.17.2',
                        'scikit-learn==1.5.2',
                        'cloudpickle==2.2.1',
                        'numpy',
                        'pandas'
                    ]}
                ]
            }
            conda_path = os.path.join(model_dir, "conda.yaml")
            with open(conda_path, 'w') as f:
                yaml.dump(conda_env, f)
            print(f"✓ conda.yaml created")
            sys.stdout.flush()
            
            # 3. Create python_env.yaml
            python_env = {
                'python': '3.9.25',
                'build_dependencies': ['pip'],
                'dependencies': [
                    'mlflow==2.17.2',
                    'scikit-learn==1.5.2',
                    'cloudpickle==2.2.1'
                ]
            }
            python_env_path = os.path.join(model_dir, "python_env.yaml")
            with open(python_env_path, 'w') as f:
                yaml.dump(python_env, f)
            print(f"✓ python_env.yaml created")
            sys.stdout.flush()
            
            # 4. Create requirements.txt
            requirements = """mlflow==2.17.2
scikit-learn==1.5.2
cloudpickle==2.2.1
"""
            req_path = os.path.join(model_dir, "requirements.txt")
            with open(req_path, 'w') as f:
                f.write(requirements)
            print(f"✓ requirements.txt created")
            sys.stdout.flush()
            
            # 5. Create MLmodel file (most important!)
            mlmodel_content = f"""artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.9.25
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.5.2
mlflow_version: 2.17.2
model_size_bytes: {file_size}
model_uuid: {mlflow.active_run().info.run_id}
run_id: {mlflow.active_run().info.run_id}
saved_input_example_info:
  artifact_path: input_example.json
  pandas_orient: split
  type: dataframe
utc_time_created: '2025-11-23 09:50:00.000000'
"""
            mlmodel_path = os.path.join(model_dir, "MLmodel")
            with open(mlmodel_path, 'w') as f:
                f.write(mlmodel_content)
            print(f"✓ MLmodel file created")
            sys.stdout.flush()
            
            # 6. Save input example
            input_example_data = X_test.iloc[:1].to_dict(orient='split')
            import json
            input_example_path = os.path.join(model_dir, "input_example.json")
            with open(input_example_path, 'w') as f:
                json.dump(input_example_data, f)
            print(f"✓ input_example.json created")
            sys.stdout.flush()
            
            # Verify all files exist
            print("\n" + "="*60)
            print("VERIFICATION")
            print("="*60)
            required_files = ["model.pkl", "MLmodel", "conda.yaml", "python_env.yaml", "requirements.txt"]
            all_good = True
            
            for filename in required_files:
                filepath = os.path.join(model_dir, filename)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"✓ {filename} ({size} bytes)")
                else:
                    print(f"❌ {filename} MISSING!")
                    all_good = False
            
            sys.stdout.flush()
            
            if not all_good:
                raise Exception("Some required files are missing!")
            
            print("\n✓✓✓ Model saved and verified successfully! ✓✓✓")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"❌ Model logging failed: {e}")
            import traceback
            traceback.print_exc()
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
