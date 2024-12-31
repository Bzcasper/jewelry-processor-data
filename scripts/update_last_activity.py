import os
import boto3
import time
import sys

def update_last_activity(bucket_name, file_name='last_activity.txt'):
    s3_client = boto3.client('s3')
    current_time = int(time.time())  # Unix timestamp in seconds
    try:
        s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=str(current_time))
        print(f"Updated last activity to {current_time}")
    except Exception as e:
        print(f"Failed to update last activity: {e}")
        sys.exit(1)

if __name__ == "__main__":
    BUCKET_NAME = os.getenv('S3_BUCKET_NAME')  # e.g., 'your-activity-bucket'
    if not BUCKET_NAME:
        print("S3_BUCKET_NAME environment variable not set.")
        sys.exit(1)
    update_last_activity(BUCKET_NAME)

