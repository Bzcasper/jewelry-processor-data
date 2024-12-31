# src/worker/maximized_worker.py

import asyncio
import io
import json
import logging
import os
import signal
from datetime import datetime
from typing import Dict, List, Optional

import aio_pika
import aioredis
import minio
import numpy as np
import pynvml  # For accurate GPU utilization monitoring
import torch
import torch.multiprocessing as mp
from elasticsearch import AsyncElasticsearch
from motor.motor_asyncio import AsyncIOMotorClient
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from PIL import Image
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    push_to_gateway,
)
from tritonclient.http import InferenceServerClient, InferInput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MaximizedWorker:
    """Enhanced worker with distributed processing capabilities."""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.setup_monitoring()
        # Initialize other components asynchronously
        asyncio.create_task(self.async_setup())

    async def async_setup(self):
        """Asynchronous setup for storage, messaging, tracing, and model client."""
        await self.setup_storage()
        await self.setup_messaging()
        self.setup_tracing()
        self.setup_model_client()
        await self.initialize_gpu_monitoring()

    def setup_monitoring(self):
        """Initialize monitoring and metrics."""
        self.registry = CollectorRegistry()
        self.metrics = {
            "processing_time": Histogram(
                "image_processing_seconds",
                "Time spent processing each image",
                registry=self.registry,
                buckets=(0.1, 0.5, 1, 2.5, 5, 10),
            ),
            "error_counter": Counter(
                "processing_errors_total",
                "Total processing errors",
                registry=self.registry,
            ),
            "gpu_utilization": Gauge(
                "gpu_utilization_percent",
                "GPU utilization percentage",
                registry=self.registry,
            ),
        }
        logger.info("Monitoring and metrics initialized.")

    async def setup_storage(self):
        """Initialize storage clients."""
        try:
            # MongoDB for metadata
            mongodb_url = os.getenv("MONGODB_URL")
            if not mongodb_url:
                raise ValueError("MONGODB_URL environment variable not set.")
            self.mongodb = AsyncIOMotorClient(mongodb_url)
            self.db = self.mongodb.jewelry_detection
            logger.info("MongoDB client initialized.")

            # MinIO for object storage
            minio_url = os.getenv("MINIO_URL")
            minio_access_key = os.getenv("MINIO_ACCESS_KEY")
            minio_secret_key = os.getenv("MINIO_SECRET_KEY")
            if not all([minio_url, minio_access_key, minio_secret_key]):
                raise ValueError("MinIO environment variables not set.")
            self.minio_client = minio.Minio(
                minio_url,
                access_key=minio_access_key,
                secret_key=minio_secret_key,
                secure=False,
            )
            logger.info("MinIO client initialized.")

            # Elasticsearch for search
            elasticsearch_url = os.getenv("ELASTICSEARCH_URL")
            if not elasticsearch_url:
                raise ValueError("ELASTICSEARCH_URL environment variable not set.")
            self.es = AsyncElasticsearch([elasticsearch_url])
            logger.info("Elasticsearch client initialized.")
        except Exception as e:
            logger.error(f"Error setting up storage: {e}")
            self.metrics["error_counter"].inc()
            raise

    async def setup_messaging(self):
        """Initialize message queues."""
        try:
            rabbitmq_url = os.getenv("RABBITMQ_URL")
            if not rabbitmq_url:
                raise ValueError("RABBITMQ_URL environment variable not set.")
            self.rabbitmq = await aio_pika.connect_robust(rabbitmq_url)
            self.channel = await self.rabbitmq.channel()
            self.queue = await self.channel.declare_queue(
                "jewelry_detection", durable=True
            )
            logger.info("RabbitMQ messaging initialized.")
        except Exception as e:
            logger.error(f"Error setting up messaging: {e}")
            self.metrics["error_counter"].inc()
            raise

    def setup_tracing(self):
        """Initialize distributed tracing."""
        try:
            tracer_provider = TracerProvider()
            jaeger_host = os.getenv("JAEGER_AGENT_HOST", "jaeger")
            jaeger_port = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_host,
                agent_port=jaeger_port,
            )
            tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            trace.set_tracer_provider(tracer_provider)
            logger.info("Distributed tracing with Jaeger initialized.")
        except Exception as e:
            logger.error(f"Error setting up tracing: {e}")
            self.metrics["error_counter"].inc()
            raise

    def setup_model_client(self):
        """Initialize Triton model client."""
        try:
            model_server_url = os.getenv("MODEL_SERVER_URL")
            if not model_server_url:
                raise ValueError("MODEL_SERVER_URL environment variable not set.")
            self.model_client = InferenceServerClient(url=model_server_url)
            logger.info("Triton Inference Server client initialized.")
        except Exception as e:
            logger.error(f"Error setting up model client: {e}")
            self.metrics["error_counter"].inc()
            raise

    async def initialize_gpu_monitoring(self):
        """Initialize GPU monitoring using pynvml."""
        try:
            pynvml.nvmlInit()
            self.gpu_handles = []
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.gpu_handles.append(handle)
            logger.info(f"GPU monitoring initialized for {device_count} GPU(s).")
        except pynvml.NVMLError as e:
            logger.error(f"Error initializing GPU monitoring: {e}")
            self.metrics["error_counter"].inc()

    async def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess the image for model inference."""
        try:
            # Example preprocessing: Convert to RGB and normalize
            image = image.convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # CxHxW
            return tensor
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            self.metrics["error_counter"].inc()
            raise

    async def process_image(self, image_data: bytes) -> Dict:
        """Process a single image with comprehensive tracking."""
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("process_image") as span:
            try:
                with self.metrics["processing_time"].time():
                    # Convert image to tensor
                    image = Image.open(io.BytesIO(image_data))
                    tensor = await self.preprocess_image(image)

                    # Get model prediction
                    prediction = await self.get_model_prediction(tensor)

                    # Post-process results
                    processed_result = self.postprocess_prediction(prediction)

                    # Store results
                    await self.store_results(processed_result)

                    logger.info(
                        f"Image processed successfully: {processed_result['id']}"
                    )
                    return processed_result

            except Exception as e:
                self.metrics["error_counter"].inc()
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                logger.error(f"Error processing image: {e}")
                raise

    async def get_model_prediction(self, tensor: torch.Tensor) -> Dict:
        """Get prediction from Triton model server."""
        try:
            inputs = [InferInput("input", tensor.shape, "FP32")]
            inputs[0].set_data_from_numpy(tensor.numpy())

            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model_client.infer("jewelry_detector", inputs)
            )
            output = response.as_numpy("output")
            logger.debug("Model prediction obtained.")
            return {"output": output}
        except Exception as e:
            logger.error(f"Error getting model prediction: {e}")
            self.metrics["error_counter"].inc()
            raise

    def postprocess_prediction(self, prediction: Dict) -> Dict:
        """Post-process model prediction."""
        try:
            # Example post-processing: Extract relevant information
            processed = {
                "id": datetime.utcnow().isoformat(),
                "prediction": prediction["output"].tolist(),
                "timestamp": datetime.utcnow().isoformat(),
            }
            logger.debug("Prediction post-processed.")
            return processed
        except Exception as e:
            logger.error(f"Error post-processing prediction: {e}")
            self.metrics["error_counter"].inc()
            raise

    async def store_results(self, result: Dict):
        """Store processing results in distributed storage."""
        try:
            # Store metadata in MongoDB
            await self.db.results.insert_one(result)
            logger.debug("Result stored in MongoDB.")

            # Store result in MinIO
            object_name = f"{result['id']}.json"
            data = json.dumps(result).encode()
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.minio_client.put_object(
                    "results",
                    object_name,
                    io.BytesIO(data),
                    length=len(data),
                    content_type="application/json",
                ),
            )
            logger.debug("Result stored in MinIO.")

            # Index in Elasticsearch
            await self.es.index(index="jewelry_detection", document=result)
            logger.debug("Result indexed in Elasticsearch.")
        except Exception as e:
            logger.error(f"Error storing results: {e}")
            self.metrics["error_counter"].inc()
            raise

    async def process_messages(self):
        """Process messages from queue."""
        logger.info("Started processing messages from queue.")
        try:
            async with self.queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        try:
                            await self.process_image(message.body)
                        except Exception as e:
                            logger.error(f"Failed to process message: {e}")
                            # Optionally, implement retry logic or move to a
                            # dead-letter queue
        except asyncio.CancelledError:
            logger.info("Message processing task cancelled.")
        except Exception as e:
            logger.error(f"Error in process_messages: {e}")
            self.metrics["error_counter"].inc()

    async def monitor_resources(self):
        """Monitor system resources."""
        logger.info("Started resource monitoring.")
        try:
            while not self.shutdown_event.is_set():
                total_util = 0
                count = len(self.gpu_handles)
                if count > 0:
                    for handle in self.gpu_handles:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        total_util += util
                    avg_util = total_util / count
                    self.metrics["gpu_utilization"].set(avg_util)
                    logger.debug(f"Average GPU Utilization: {avg_util}%")
                else:
                    self.metrics["gpu_utilization"].set(0)
                    logger.debug("No GPUs available for utilization monitoring.")
                await asyncio.sleep(15)
        except asyncio.CancelledError:
            logger.info("Resource monitoring task cancelled.")
        except Exception as e:
            logger.error(f"Error in monitor_resources: {e}")
            self.metrics["error_counter"].inc()

    async def run(self):
        """Run the worker with all monitoring tasks."""
        logger.info("MaximizedWorker is starting.")
        # Handle graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig, lambda: asyncio.create_task(self.shutdown(sig))
            )

        tasks = [
            asyncio.create_task(self.process_messages()),
            asyncio.create_task(self.monitor_resources()),
            asyncio.create_task(self.push_metrics_periodically()),
        ]

        await self.shutdown_event.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.cleanup()
        logger.info("MaximizedWorker has shut down.")

    async def push_metrics_periodically(self):
        """Push Prometheus metrics to the Pushgateway periodically."""
        pushgateway_url = os.getenv("PROMETHEUS_PUSHGATEWAY_URL")
        if not pushgateway_url:
            logger.warning(
                "PROMETHEUS_PUSHGATEWAY_URL not set. Metrics will not be pushed."
            )
            return

        interval = 60  # seconds
        try:
            while not self.shutdown_event.is_set():
                try:
                    push_to_gateway(
                        pushgateway_url, job="maximized_worker", registry=self.registry
                    )
                    logger.debug("Metrics pushed to Pushgateway.")
                except Exception as e:
                    logger.error(f"Failed to push metrics: {e}")
                    self.metrics["error_counter"].inc()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Metrics pushing task cancelled.")
        except Exception as e:
            logger.error(f"Error in push_metrics_periodically: {e}")
            self.metrics["error_counter"].inc()

    async def shutdown(self, sig):
        """Handle graceful shutdown."""
        logger.info(f"Received exit signal {sig.name}...")
        self.shutdown_event.set()

    async def cleanup(self):
        """Cleanup resources before shutdown."""
        try:
            await self.rabbitmq.close()
            await self.es.close()
            self.mongodb.close()
            pynvml.nvmlShutdown()
            logger.info("All resources have been cleaned up.")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def close_connections(self):
        """Close all connections gracefully."""
        await self.cleanup()


# Entry point for the worker
async def main():
    """Initialize and run the maximized worker."""
    worker = MaximizedWorker()
    await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("MaximizedWorker interrupted and is shutting down.")
