import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import requests
import yaml
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect


@dataclass
class ValidationMetrics:
    """Deployment validation metrics."""

    pod_health: bool
    resource_usage: Dict[str, float]
    api_latency: float
    error_rate: float
    gpu_utilization: float


class EnhancedClusterManager:
    """Enhanced cluster management with graceful shutdown and validation."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.namespace = "jewelry-processor"
        self.kube_client = self._init_kubernetes()
        self.prom_client = PrometheusConnect(url="http://prometheus-service:9090")

    def _init_kubernetes(self) -> client.CoreV1Api:
        """Initialize Kubernetes client."""
        config.load_kube_config()
        return client.CoreV1Api()

    def _load_config(self, path: str) -> Dict:
        """Load configuration from yaml file."""
        with open(path) as f:
            return yaml.safe_load(f)

    def validate_deployment(self) -> Tuple[bool, str]:
        """Comprehensive deployment validation."""
        try:
            # Collect validation metrics
            metrics = self._collect_validation_metrics()

            # Check deployment health
            if not metrics.pod_health:
                return False, "Pod health check failed"

            # Validate resource usage
            if (
                metrics.resource_usage["cpu"] > 90
                or metrics.resource_usage["memory"] > 90
            ):
                return False, "Resource usage too high"

            # Check API responsiveness
            if metrics.api_latency > 1000:  # 1 second
                return False, "API latency too high"

            # Verify error rate
            if metrics.error_rate > 0.01:  # 1% error rate
                return False, "Error rate too high"

            # Check GPU utilization
            if metrics.gpu_utilization < 10:  # Under-utilized
                return False, "GPU under-utilized"

            return True, "Deployment validation successful"

        except Exception as e:
            return False, f"Validation failed: {str(e)}"

    def _collect_validation_metrics(self) -> ValidationMetrics:
        """Collect comprehensive validation metrics."""
        # Check pod health
        pods = self.kube_client.list_namespaced_pod(self.namespace)
        pod_health = all(pod.status.phase == "Running" for pod in pods.items)

        # Get resource usage
        resource_usage = {
            "cpu": float(
                self.prom_client.custom_query("avg(container_cpu_usage_seconds_total)")[0]["value"][1]
            ),
            "memory": float(
                self.prom_client.custom_query("avg(container_memory_usage_bytes)")[0]["value"][1]
            ),
        }

        # Check API latency
        api_latency = float(
            self.prom_client.custom_query(
                "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
            )[0]["value"][1]
        )

        # Get error rate
        error_rate = float(
            self.prom_client.custom_query(
                'sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))'
            )[0]["value"][1]
        )

        # Check GPU utilization
        gpu_utilization = float(
            self.prom_client.custom_query("avg(nvidia_gpu_utilization)")[0]["value"][1]
        )

        return ValidationMetrics(
            pod_health=pod_health,
            resource_usage=resource_usage,
            api_latency=api_latency,
            error_rate=error_rate,
            gpu_utilization=gpu_utilization,
        )

    def graceful_shutdown(self):
        """Enhanced graceful shutdown procedure."""
        try:
            print("Initiating graceful shutdown...")

            # 1. Stop accepting new requests
            self._update_ingress(enabled=False)

            # 2. Wait for existing jobs to complete
            self._wait_for_job_completion()

            # 3. Scale down web servers
            self._scale_deployment("jewelry-processor-web", 0)

            # 4. Scale down workers gradually
            self._gradual_worker_shutdown()

            # 5. Backup final state
            self._backup_final_state()

            # 6. Clean up resources
            self._cleanup_resources()

            print("Shutdown completed successfully!")

        except Exception as e:
            print(f"Shutdown failed: {str(e)}")
            raise

    def _update_ingress(self, enabled: bool):
        """Update ingress to stop/start accepting traffic."""
        networking_v1 = client.NetworkingV1Api()

        try:
            ingress = networking_v1.read_namespaced_ingress(
                "jewelry-processor-ingress", self.namespace
            )

            if not enabled:
                # Add annotation to reject new connections
                annotations = ingress.metadata.annotations or {}
                annotations["nginx.ingress.kubernetes.io/server-snippet"] = "return 503;"
                ingress.metadata.annotations = annotations

            else:
                # Remove the annotation if enabling
                annotations = ingress.metadata.annotations or {}
                annotations.pop("nginx.ingress.kubernetes.io/server-snippet", None)
                ingress.metadata.annotations = annotations

            networking_v1.replace_namespaced_ingress(
                "jewelry-processor-ingress", self.namespace, ingress
            )

        except Exception as e:
            print(f"Failed to update ingress: {str(e)}")
            raise

    def _wait_for_job_completion(self, timeout: int = 300):
        """Wait for all processing jobs to complete."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check Redis queue length
            queue_length = self._get_queue_length()
            if queue_length == 0:
                return
            print(f"Current queue length: {queue_length}. Waiting...")
            time.sleep(10)
        raise TimeoutError("Jobs failed to complete within timeout")

    def _gradual_worker_shutdown(self):
        """Gradually scale down worker pods."""
        try:
            current_replicas = self._get_deployment_replicas("jewelry-processor-worker")

            for replicas in range(current_replicas - 1, -1, -1):
                self._scale_deployment("jewelry-processor-worker", replicas)
                print(f"Scaled 'jewelry-processor-worker' to {replicas} replicas.")
                time.sleep(30)  # Wait between scaling steps

        except Exception as e:
            print(f"Failed to gradually shutdown workers: {str(e)}")
            raise

    def _backup_final_state(self):
        """Backup final state before shutdown."""
        try:
            # Backup Redis data
            self._exec_pod_command("redis-0", ["redis-cli", "save"])

            # Export Prometheus metrics
            self._backup_metrics()

            # Backup MongoDB data
            self._backup_mongodb()

        except Exception as e:
            print(f"Failed to backup state: {str(e)}")
            raise

    def _cleanup_resources(self):
        """Clean up cluster resources."""
        try:
            # Delete deployments
            subprocess.run(
                ["kubectl", "delete", "deployment", "--all", "-n", self.namespace],
                check=True,
            )
            print("All deployments deleted.")

            # Delete services
            subprocess.run(
                ["kubectl", "delete", "service", "--all", "-n", self.namespace],
                check=True,
            )
            print("All services deleted.")

            # Delete PVCs
            subprocess.run(
                ["kubectl", "delete", "pvc", "--all", "-n", self.namespace],
                check=True,
            )
            print("All PersistentVolumeClaims deleted.")

            # Delete namespace
            subprocess.run(
                ["kubectl", "delete", "namespace", self.namespace], check=True
            )
            print(f"Namespace '{self.namespace}' deleted.")

        except subprocess.CalledProcessError as e:
            print(f"Failed to cleanup resources: {e}")
            raise

    def _scale_deployment(self, deployment_name: str, replicas: int):
        """Scale a deployment to the specified number of replicas."""
        apps_v1 = client.AppsV1Api()
        body = {"spec": {"replicas": replicas}}
        apps_v1.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=self.namespace,
            body=body
        )
        print(f"Deployment '{deployment_name}' scaled to {replicas} replicas.")

    def _get_deployment_replicas(self, deployment_name: str) -> int:
        """Get current replica count for deployment."""
        apps_v1 = client.AppsV1Api()
        deployment = apps_v1.read_namespaced_deployment(
            deployment_name,
            self.namespace
        )
        return deployment.spec.replicas or 0

    def _get_queue_length(self) -> int:
        """Get current queue length from Redis."""
        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "exec",
                    "-n",
                    self.namespace,
                    "redis-0",
                    "--",
                    "redis-cli",
                    "llen",
                    "processing_queue"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            queue_length = int(result.stdout.strip())
            print(f"Retrieved queue length: {queue_length}")
            return queue_length
        except subprocess.CalledProcessError as e:
            print(f"Error retrieving queue length: {e}")
            return 0
        except ValueError:
            print("Invalid queue length received.")
            return 0

    def _exec_pod_command(self, pod_name: str, command: List[str]):
        """Execute a command inside a pod."""
        try:
            subprocess.run(
                ["kubectl", "exec", "-n", self.namespace, pod_name, "--"] + command,
                check=True
            )
            print(f"Executed command in pod '{pod_name}': {' '.join(command)}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute command in pod '{pod_name}': {e}")
            raise

    def _backup_metrics(self):
        """Backup Prometheus metrics."""
        try:
            metrics = self.prom_client.get_current_metric_value(
                metric_name="http_requests_total"
            )
            with open("metrics_backup.json", "w") as f:
                json.dump(metrics, f)
            print("Prometheus metrics backed up to 'metrics_backup.json'.")
        except Exception as e:
            print(f"Failed to backup Prometheus metrics: {e}")
            raise

    def _backup_mongodb(self):
        """Backup MongoDB data."""
        try:
            subprocess.run(
                ["kubectl", "exec", "-n", self.namespace, "mongodb-0", "--", "mongodump", "--out", "/backup"],
                check=True
            )
            subprocess.run(
                ["kubectl", "cp", f"{self.namespace}/mongodb-0:/backup", "./mongodb_backup"],
                check=True
            )
            print("MongoDB data backed up to './mongodb_backup'.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to backup MongoDB data: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Manage Kubernetes deployments.")
    parser.add_argument("action", choices=["validate", "shutdown"], help="Action to perform")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")

    args = parser.parse_args()

    manager = EnhancedClusterManager(args.config)

    if args.action == "validate":
        success, message = manager.validate_deployment()
        print(f"Validation result: {message}")
        sys.exit(0 if success else 1)
    elif args.action == "shutdown":
        manager.graceful_shutdown()


if __name__ == "__main__":
    main()
