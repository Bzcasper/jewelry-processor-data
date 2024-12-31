import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests
import yaml
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect


@dataclass # (1)
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
                self.prom_client.custom_query("avg(container_cpu_usage_seconds_total)")[
                    0
                ]["value"][1]
            ),
            "memory": float(
                self.prom_client.custom_query("avg(container_memory_usage_bytes)")[0][
                    "value"
                ][1]
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
                ingress.metadata.annotations[
                    "nginx.ingress.kubernetes.io/server-snippet"
                ] = "return 503;"

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
            time.sleep(10)
        raise TimeoutError("Jobs failed to complete within timeout")

    def _gradual_worker_shutdown(self):
        """Gradually scale down worker pods."""
        try:
            current_replicas = self._get_deployment_replicas("jewelry-processor-worker")

            for replicas in range(current_replicas - 1, -1, -1):
                self._scale_deployment("jewelry-processor-worker", replicas)
                time.sleep(30)  # Wait between scaling steps

        except Exception as e:
            print(f"Failed to gradually shutdown workers: {str(e)}")
            raise

    def _backup_final_state(self):
        """Backup final state before shutdown."""
        try:
            # Backup Redis data
            self._exec_pod_command("redis", ["redis-cli", "save"])

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

            # Delete services
            subprocess.run(
                ["kubectl", "delete", "service", "--all", "-n", self.namespace],
                check=True,
            )

            # Delete PVCs
            subprocess.run(
                ["kubectl", "delete", "pvc", "--all", "-n", self.namespace], check=True
            )

            # Delete namespace
            subprocess.run(
                ["kubectl", "delete", "namespace", self.namespace], check=True
            )

        except Exception as e:
            print(f"Failed to cleanup resources: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["validate", "shutdown"])
    parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args()

    manager = EnhancedClusterManager(args.config)

    if args.action == "validate":
        success, message = manager.validate_deployment()
        print(f"Validation result: {message}")
        sys.exit(0 if success else 1)

    else:
        manager.graceful_shutdown()


if __name__ == "__main__":
    main()
