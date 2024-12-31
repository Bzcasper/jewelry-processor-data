variable "project_id" {
  description = "The ID of the GCP project"
  type        = string
}

variable "region" {
  description = "The region for GCP resources"
  type        = string
  default     = "us-central1"
}

variable "instance_name" {
  description = "The name of the Compute Engine instance"
  type        = string
  default     = "jewelry-processor-instance"
}
