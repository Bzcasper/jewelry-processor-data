terraform {
  backend "gcs" {
    bucket = "data-bucket-aitoolpool
"
    prefix = "terraform/state"
  }
}
