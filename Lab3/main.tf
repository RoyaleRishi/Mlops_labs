 terraform {
        required_providers {
            google = {
                source  = "hashicorp/google"
                version = "~> 7.0"
            }
            random = {
                source  = "hashicorp/random"
                version = "~> 3.5"
            }
        }
    }

# Variables for reusability
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-east1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = "${var.region}-b"
}

# VM instance for ML inference API
resource "google_compute_instance" "ml_inference_vm" {
  name         = "ml-inference-api-${var.environment}"
  machine_type = "e2-standard-4" # Upgrade
  zone         = "${var.region}-b"

  tags = ["ml-api", "http-server", "https-server"]

  labels = {
    environment = var.environment
    purpose     = "ml-inference"
    managed_by  = "terraform"
    version     = "v2" 
  }

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30 # Upgrade
      type  = "pd-ssd" # Upgrade
    }
  }

  # Startup script to install dependencies
  metadata_startup_script = <<-EOF
            #!/bin/bash
            apt-get update
            apt-get install -y python3-pip python3-venv nginx
            pip3 install fastapi uvicorn scikit-learn pandas numpy
            systemctl start docker
            systemctl enable docker
            echo "ML inference VM initialized" > /var/log/startup.log
        EOF

  network_interface {
    network = "default"
    access_config {
      # Ephemeral external IP
    }
  }

  service_account {
    scopes = ["cloud-platform"]
  }
}

# Output the VM's external IP
output "vm_external_ip" {
  description = "External IP of ML inference VM"
  value       = google_compute_instance.ml_inference_vm.network_interface[0].access_config[0].nat_ip
}


# Cloud Storage bucket for ML models
    resource "google_storage_bucket" "model_storage" {
        name          = "${var.project_id}-ml-models-${var.environment}"
        location      = var.region
        force_destroy = true
        
        uniform_bucket_level_access = true

        labels = {
            environment = var.environment
            purpose     = "model-storage"
            managed_by  = "terraform"
        }

        versioning {
            enabled = true  # Keep model version history
        }

        lifecycle_rule {
            condition {
                age = 90  # Days
            }
            action {
                type = "Delete"
            }
        }
    }

    # Create a sample folder structure
    resource "google_storage_bucket_object" "models_folder" {
        name    = "models/"
        content = " "
        bucket  = google_storage_bucket.model_storage.name
    }

    resource "google_storage_bucket_object" "datasets_folder" {
        name    = "datasets/"
        content = " "
        bucket  = google_storage_bucket.model_storage.name
    }

    output "bucket_name" {
        description = "Name of the model storage bucket"
        value       = google_storage_bucket.model_storage.name
    }

    # Firewall rule to allow HTTP traffic
    resource "google_compute_firewall" "allow_ml_api" {
        name    = "allow-ml-api-${var.environment}"
        network = "default"

        allow {
            protocol = "tcp"
            ports    = ["8000", "80", "443"]  # FastAPI typically uses 8000
        }

        source_ranges = ["0.0.0.0/0"]  # In production, restrict this!
        target_tags   = ["ml-api"]

        description = "Allow HTTP/HTTPS traffic to ML inference API"
    }

    output "api_endpoint" {
        description = "ML API endpoint"
        value       = "http://${google_compute_instance.ml_inference_vm.network_interface[0].access_config[0].nat_ip}:8000"
    }



     # Random suffix for unique instance name
    resource "random_id" "db_suffix" {
        byte_length = 4
    }

    # Cloud SQL instance for prediction logging
    resource "google_sql_database_instance" "prediction_logs" {
        name             = "ml-predictions-${var.environment}-${random_id.db_suffix.hex}"
        database_version = "POSTGRES_15"
        region           = var.region
        
        settings {
            tier = "db-f1-micro"  # Smallest instance for dev
            
            ip_configuration {
                ipv4_enabled = true
                authorized_networks {
                    name  = "allow-all"  # In production, restrict this!
                    value = "0.0.0.0/0"
                }
            }

            backup_configuration {
                enabled = true
                start_time = "03:00"
            }
        }

        deletion_protection = false  # For easier cleanup in dev
    }

    # Create database
    resource "google_sql_database" "predictions" {
        name     = "predictions"
        instance = google_sql_database_instance.prediction_logs.name
    }

    # Create database user
    resource "google_sql_user" "ml_api_user" {
        name     = "ml_api"
        instance = google_sql_database_instance.prediction_logs.name
        password = "changeme123"  # In production, use secrets management!
    }

    output "database_connection" {
        description = "Database connection details"
        value = {
            host     = google_sql_database_instance.prediction_logs.public_ip_address
            database = google_sql_database.predictions.name
            user     = google_sql_user.ml_api_user.name
        }
        sensitive = false
    }