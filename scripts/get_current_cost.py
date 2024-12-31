# scripts/get_current_cost.py

from google.cloud import billing_v1
import os
import sys

def get_project_billing_info(project_id):
    client = billing_v1.CloudBillingClient()
    project_name = f"projects/{project_id}"
    billing_info = client.get_project_billing_info(name=project_name)
    return billing_info

def get_current_cost(project_id):
    # Placeholder for actual cost retrieval logic
    # Implement billing data retrieval, possibly from BigQuery if billing export is enabled
    # For demonstration, returning a fixed value
    return 50  # Replace with actual cost retrieval

def main():
    project_id = os.getenv('PROJECT')
    if not project_id:
        print("PROJECT environment variable not set.", file=sys.stderr)
        sys.exit(1)
    
    billing_info = get_project_billing_info(project_id)
    if billing_info.billing_enabled:
        cost = get_current_cost(project_id)
    else:
        cost = 0
    
    print(cost)

if __name__ == "__main__":
    main()
