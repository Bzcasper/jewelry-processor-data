import subprocess
import argparse
import sys

def create_namespace(namespace):
    try:
        subprocess.run(['kubectl', 'create', 'namespace', namespace], check=True)
        print(f"Namespace '{namespace}' created.")
    except subprocess.CalledProcessError as e:
        if "already exists" in str(e):
            print(f"Namespace '{namespace}' already exists.")
        else:
            print(f"Failed to create namespace '{namespace}': {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Create Kubernetes namespace if it does not exist.")
    parser.add_argument("--namespace", required=True, help="Namespace to create.")
    args = parser.parse_args()
    create_namespace(args.namespace)

if __name__ == "__main__":
    main()
