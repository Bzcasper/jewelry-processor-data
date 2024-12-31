import argparse
import boto3
import sys

def get_last_activity(bucket_name, file_name='last_activity.txt'):
    s3_client = boto3.client('s3')
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
        last_activity = int(response['Body'].read().decode('utf-8'))
        print(last_activity)
    except Exception as e:
        print(f"Failed to retrieve last activity: {e}")
        # Return a high value to force shutdown
        print(0)
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Retrieve last activity timestamp from S3.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name.")
    parser.add_argument("--file", required=False, default='last_activity.txt', help="File name to retrieve.")

    args = parser.parse_args()

    get_last_activity(args.bucket, args.file)

if __name__ == "__main__":
    main()
