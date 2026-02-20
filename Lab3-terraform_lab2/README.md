# Lab 3: Terraform Infrastructure Deployment

## Modifications from Original Lab

Instead of deploying a simple VM, I created a complete ML inference API infrastructure with multiple interconnected resources:

### Key Changes:
1. **Different machine type**: `e2-standard-2` (2 vCPUs, 8GB RAM) instead of `f1-micro`
2. **Different OS**: Ubuntu 22.04 LTS instead of Debian 11
3. **Additional resources added**:
   - Cloud Storage bucket for ML model artifacts
   - Cloud SQL PostgreSQL database for prediction logging
   - Firewall rules for API access
   - Folder structure in storage bucket
4. **Used variables** for project_id, region, and environment (original lab had hardcoded values)
5. **Added outputs** to display VM IP, bucket name, database connection info, and API endpoint
6. **Included startup script** to auto-install ML dependencies (FastAPI, scikit-learn, pandas, numpy)

## Setup Steps

### 1. Authentication
Since service account key creation was blocked by organization policy, I used Application Default Credentials:

```bash
gcloud auth application-default login
gcloud config set project project-id
```

### 2. Enable Required APIs
```bash
gcloud services enable compute.googleapis.com storage.googleapis.com sqladmin.googleapis.com
```

### 3. Create Terraform Configuration Files

**Created `main.tf`** with:
- Provider configuration (Google, Random)
- Variable declarations (project_id, region, environment)
- VM instance resource
- Cloud Storage bucket with versioning and lifecycle rules
- Firewall rule for HTTP/HTTPS traffic
- Cloud SQL instance, database, and user
- Output definitions

**Created `terraform.tfvars`**:
```hcl
project_id  = "project-id"
region      = "us-east1"
environment = "dev"
```

### 4. Initialize Terraform
```bash
terraform init -upgrade
```

### 5. Preview Changes
```bash
terraform plan
```

### 6. Deploy Infrastructure
```bash
terraform apply
```

## Resources Created

1. **VM Instance**: `ml-inference-api-dev` (e2-standard-2, us-east1-b)
2. **Storage Bucket**: `project-id-ml-models-dev` with versioning enabled
3. **Cloud SQL**: PostgreSQL 15 instance (db-f1-micro)
4. **Database**: `predictions` database
5. **Database User**: `ml_api` user
6. **Firewall Rule**: Allows traffic on ports 80, 443, 8000

## Cleanup

To destroy all resources and avoid charges:
```bash
terraform destroy
```
