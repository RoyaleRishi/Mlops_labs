import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from google.cloud import storage
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_gcs_path', type=str, required=True,
                        help='GCS path to training data CSV')
    parser.add_argument('--model_dir', type=str, default=os.getenv('AIP_MODEL_DIR'),
                        help='Directory to save model')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in random forest')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Max depth of trees')
    parser.add_argument('--min_samples_split', type=int, default=2,
                        help='Min samples required to split node')
    return parser.parse_args()

def load_data(gcs_path):
    """Load data from GCS"""
    print(f"Loading data from: {gcs_path}")
    df = pd.read_csv(gcs_path)
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def preprocess_data(df):
    print("\nPreprocessing data...")
    print(f"Initial shape: {df.shape}")
    
    # Check for missing values
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    
    # Handle missing values (if any)
    df = df.dropna()
    print(f"Shape after dropping NaN: {df.shape}")
    
    # Check target distribution
    print(f"\nTarget distribution:")
    print(df['target'].value_counts())
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {X.columns.tolist()}")
    
    return X, y

def train_model(X_train, y_train, n_estimators, max_depth, min_samples_split):
    """Train Random Forest model"""
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    print(f"Hyperparameters:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - max_depth: {max_depth}")
    print(f"  - min_samples_split: {min_samples_split}")
    print(f"  - random_state: 42")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    print("\nModel training complete!")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, 
                                target_names=['No Disease', 'Disease']))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    print(f"True Negatives: {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives: {cm[1,1]}")
    
    return accuracy

def save_model(model, model_dir):
    """Save model to GCS"""
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    
    model_filename = 'model.joblib'
    local_path = model_filename
    
    # Save locally first
    joblib.dump(model, local_path)
    print(f"Model saved locally to: {local_path}")
    
    # Upload to GCS if model_dir is provided
    if model_dir:
        gcs_path = os.path.join(model_dir, model_filename)
        print(f"Uploading model to: {gcs_path}")
        
        try:
            # Parse GCS path
            path_parts = gcs_path.replace('gs://', '').split('/')
            bucket_name = path_parts[0]
            blob_path = '/'.join(path_parts[1:])
            
            # Upload
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            
            print(f"✓ Model uploaded successfully to gs://{bucket_name}/{blob_path}")
        except Exception as e:
            print(f"✗ Error uploading model: {e}")
            raise
    else:
        print("No model_dir provided, model saved locally only")

def main():
    """Main training pipeline"""
    print("="*50)
    print("HEART DISEASE PREDICTION - TRAINING PIPELINE")
    print("="*50)
    
    # Parse arguments
    args = get_args()
    
    print(f"\nConfiguration:")
    print(f"  - Data path: {args.data_gcs_path}")
    print(f"  - Model output dir: {args.model_dir}")
    print(f"  - n_estimators: {args.n_estimators}")
    print(f"  - max_depth: {args.max_depth}")
    print(f"  - min_samples_split: {args.min_samples_split}")
    
    # Load and preprocess data
    df = load_data(args.data_gcs_path)
    X, y = preprocess_data(df)
    
    # Split data
    print("\n" + "="*50)
    print("SPLITTING DATA")
    print("="*50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training set class distribution:")
    print(y_train.value_counts())
    print(f"Test set class distribution:")
    print(y_test.value_counts())
    
    # Train model
    model = train_model(
        X_train, y_train, 
        args.n_estimators, 
        args.max_depth,
        args.min_samples_split
    )
    
    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, args.model_dir)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == '__main__':
    main()