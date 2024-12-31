import argparse
import subprocess
import sys

def cluster_exists(project, region, cluster, zone=None):
    try:
        cmd = ["gcloud", "container", "clusters", "describe", cluster, "--project", project, "--region", region]
        if zone:
            cmd.extend(["--zone", zone])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def create_cluster(project, region, cluster, zone=None):
    print(f"Creating GKE cluster '{cluster}' in project '{project}', region '{region}'...")
    cmd = [
        "gcloud", "container", "clusters", "create", cluster,
        "--project", project,
        "--region", region,
        "--num-nodes", "3",
        "--machine-type", "e2-medium",
        "--enable-autoscaling",
        "--min-nodes", "1",
        "--max-nodes", "5",
        "--enable-autorepair",
        "--enable-autoupgrade"
    ]
    if zone:
        cmd.extend(["--zone", zone])
    subprocess.run(cmd, check=True)
    print(f"GKE cluster '{cluster}' created successfully.")

def main():
    parser = argparse.ArgumentParser(description="Check and create GKE cluster if it does not exist.")
    parser.add_argument("--project", required=True, help="Google Cloud project ID.")
    parser.add_argument("--region", required=True, help="GKE cluster region.")
    parser.add_argument("--cluster", required=True, help="GKE cluster name.")
    parser.add_argument("--zone", required=False, help="GKE cluster zone.")

    args = parser.parse_args()

    exists = cluster_exists(args.project, args.region, args.cluster, args.zone)
    if exists:
        print(f"GKE cluster '{args.cluster}' already exists in project '{args.project}', region '{args.region}'.")
    else:
        create_cluster(args.project, args.region, args.cluster, args.zone)

if __name__ == "__main__":
    main()
