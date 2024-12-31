# scripts/cleanup_resources.py
import boto3
import subprocess
import sys

def cleanup_s3_buckets():
    s3 = boto3.client('s3')
    response = s3.list_buckets()
    for bucket in response['Buckets']:
        try:
            tags = s3.get_bucket_tagging(Bucket=bucket['Name'])
            for tag in tags.get('TagSet', []):
                if tag['Key'] == 'auto-cleanup' and tag['Value'] == 'true':
                    s3.delete_bucket(Bucket=bucket['Name'])
                    print(f"Deleted S3 bucket: {bucket['Name']}")
        except s3.exceptions.NoSuchTagSet:
            continue
        except Exception as e:
            print(f"Error deleting bucket '{bucket['Name']}': {e}")

def cleanup_gke_clusters(project, region, zone, cluster):
    try:
        subprocess.run(["gcloud", "container", "clusters", "delete", cluster, "--quiet",
                        "--region", region, "--project", project, "--zone", zone], check=True)
        print(f"Deleted GKE cluster: {cluster}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to delete GKE cluster '{cluster}': {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 5:
        print("Usage: python cleanup_resources.py <project> <region> <zone> <cluster>")
        sys.exit(1)
    
    project = sys.argv[1]
    region = sys.argv[2]
    zone = sys.argv[3]
    cluster = sys.argv[4]

    # Cleanup S3 Buckets
    cleanup_s3_buckets()

    # Cleanup GKE Clusters
    cleanup_gke_clusters(project, region, zone, cluster)

if __name__ == "__main__":
    main()
