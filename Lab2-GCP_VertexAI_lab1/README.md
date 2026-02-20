## Project Overview

End-to-end MLOps pipeline for heart disease prediction using Google Cloud Vertex AI with custom scikit-learn model.

## Dataset

**Source:** [UCI Heart Disease Dataset - Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**Description:** 
- 303 patient records
- 13 clinical features
- Binary classification (disease present/absent)
- Cleveland database subset

**Features:**
- `age`: Age in years
- `sex`: Gender (1=male, 0=female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting ECG results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina
- `oldpeak`: ST depression
- `slope`: ST segment slope (0-2)
- `ca`: Number of major vessels (0-3)
- `thal`: Thalassemia (1-3)
- `target`: Disease diagnosis (0/1)

## Model Details

**Algorithm:** Random Forest Classifier

**Hyperparameters:**
- `n_estimators`: 150 trees
- `max_depth`: 15 levels
- `min_samples_split`: 5 samples
- `random_state`: 42

**Framework:** scikit-learn 0.23.2

### Training Pipeline

1. **Data Preprocessing**
   - No missing values in dataset
   - Features already normalized
   - Stratified train-test split (80-20)

2. **Model Training**
   - Custom Python package structure
   - Pre-built scikit-learn container
   - Cloud Storage for artifacts

3. **Model Registry**
   - Versioned model artifacts
   - Model metadata tracking
   - Easy rollback capability

4. **Deployment**
   - Online prediction endpoint
   - Auto-scaling (1-1 nodes)
   - Low latency (<100ms)

### Project Structure
vertex-ai-heart-disease/
├── README.md
├── screenshots/
├── trainer/
│   ├── init.py
│   └── task.py          # Training script
├── setup.py              # Package configuration

## Reproduction Steps

### Prerequisites
- GCP account with billing enabled
- Vertex AI API enabled
- ~$5-10 in GCP credits

### Setup
```bash
# Set environment variables
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export BUCKET_PREFIX="your-name"

# Create buckets
gsutil mb -l $REGION gs://${BUCKET_PREFIX}-heart-dataset
gsutil mb -l $REGION gs://${BUCKET_PREFIX}-heart-models
gsutil mb -l $REGION gs://${BUCKET_PREFIX}-heart-source

# Upload dataset
gsutil cp heart.csv gs://${BUCKET_PREFIX}-heart-dataset/

# Build and upload package
python setup.py sdist --formats=gztar
gsutil cp dist/trainer-0.1.tar.gz gs://${BUCKET_PREFIX}-heart-source/
```

### Training

Create custom training job in Vertex AI with:
- Container: Pre-built scikit-learn 0.23
- Package: `gs://${BUCKET_PREFIX}-heart-source/lab2-0.1.0.tar.gz`
- Module: `trainer.task`
- Arguments:
--data_gcs_path=gs://${BUCKET_PREFIX}-heart-dataset/heart.csv
--n_estimators=150
--max_depth=15
--min_samples_split=5

### Deployment

1. Import model from `gs://${BUCKET_PREFIX}-heart-models/output/model.joblib`
2. Create endpoint
3. Deploy model with `n1-standard-2` machine

## Modifications from Original Lab

1. **Different Dataset:** Heart Disease (UCI) vs Stroke dataset
2. **Enhanced Preprocessing:** Added detailed data exploration
3. **Additional Hyperparameters:** min_samples_split for fine-tuning
4. **Feature Importance:** Implemented feature analysis
5. **Detailed Logging:** Comprehensive training logs
6. **Evaluation Metrics:** Added confusion matrix and classification report

## Cleanup
```bash
# Delete endpoint
gcloud ai endpoints delete ENDPOINT_ID --region=us-central1

# Delete model
gcloud ai models delete MODEL_ID --region=us-central1

# Delete buckets
gsutil rm -r gs://${BUCKET_PREFIX}-heart-dataset
gsutil rm -r gs://${BUCKET_PREFIX}-heart-models
gsutil rm -r gs://${BUCKET_PREFIX}-heart-source
