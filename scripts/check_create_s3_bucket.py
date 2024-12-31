import argparse
import boto3
from botocore.exceptions import ClientError
import sys

def bucket_exists(s3_client, bucket_name):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            return False
        else:
            raise e

def create_bucket(s3_client, bucket_name, region='us-east-1'):
    print(f"Creating S3 bucket '{bucket_name}' in region '{region}'...")
    try:
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
        print(f"S3 bucket '{bucket_name}' created successfully.")
    except ClientError as e:
        print(f"Failed to create S3 bucket '{bucket_name}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Check and create S3 bucket if it does not exist.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name.")
    parser.add_argument("--region", required=False, default='us-east-1', help="AWS region for the bucket.")

    args = parser.parse_args()

    s3_client = boto3.client('s3', region_name=args.region)

    try:
        exists = bucket_exists(s3_client, args.bucket)
    except ClientError as e:
        print(f"Error checking bucket existence: {e}")
        sys.exit(1)

    if exists:
        print(f"S3 bucket '{args.bucket}' already exists.")
    else:
        create_bucket(s3_client, args.bucket, args.region)

if __name__ == "__main__":
    main()
