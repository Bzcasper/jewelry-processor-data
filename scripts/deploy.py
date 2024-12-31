import argparse
import subprocess
import sys
import yaml

def deploy_application(config_path):
    try:
        print("Deploying application...")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Apply all Kubernetes resources
        for resource in config.get('resources', []):
            cmd = ["kubectl", "apply", "-f", resource]
            subprocess.run(cmd, check=True)
            print(f"Applied resource: {resource}")
        
        print("Application deployed successfully.")
    except Exception as e:
        print(f"Deployment failed: {e}")
        sys.exit(1)

def shutdown_application(config_path):
    try:
        print("Shutting down application...")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Scale down all specified deployments to zero replicas
        for deployment in config.get('deployments', []):
            cmd = ["kubectl", "scale", "deployment", deployment, "--replicas=0"]
            subprocess.run(cmd, check=True)
            print(f"Scaled deployment '{deployment}' to zero replicas.")
        
        print("Application shut down successfully.")
    except Exception as e:
        print(f"Shutdown failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Deploy or Shutdown the application.")
    parser.add_argument("action", choices=["deploy", "shutdown"], help="Action to perform.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")

    args = parser.parse_args()

    if args.action == "deploy":
        deploy_application(args.config)
    elif args.action == "shutdown":
        shutdown_application(args.config)
    else:
        print("Invalid action. Use 'deploy' or 'shutdown'.")
        sys.exit(1)

if __name__ == "__main__":
    main()

