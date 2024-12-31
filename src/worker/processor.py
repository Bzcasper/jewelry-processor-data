# src/worker/processor.py

import asyncio
import io
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import aiohttp
import aioredis
import albumentations as A
import cloudinary
import cloudinary.uploader
import cv2
import numpy as np
import torch
import yaml
from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image, ImageEnhance, ImageFilter
from prometheus_client import Counter, Gauge, Histogram
from ray import init as ray_init
from sentence_transformers import SentenceTransformer
from torchvision.transforms import functional as TF
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

from models.detector import DetectionResult, JewelryDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing with enhanced metadata."""

    input_paths: List[Path]
    output_dir: Path
    batch_size: int = 32
    generate_embeddings: bool = True
    extract_metadata: bool = True
    detect_objects: bool = True
    generate_tags: bool = True
    export_formats: List[str] = field(
        default_factory=lambda: ["csv", "json", "parquet"]
    )
    cloudinary_config: Optional[Dict] = None


@dataclass
class ProcessingResult:
    image_id: str
    detections: DetectionResult
    metadata: Dict
    processed_url: str
    processing_time: float


@dataclass
class ProcessingPipeline:
    """Configurable image processing pipeline."""

    steps: List[Dict[str, Union[str, float, Dict]]] = field(default_factory=list)
    batch_size: int = 16
    enable_gpu: bool = True
    use_ray: bool = False


class ProcessingMetrics:
    """Metrics collection for monitoring."""

    def __init__(self):
        self.processing_time = Histogram(
            "image_processing_seconds",
            "Time spent processing each image",
            buckets=(1, 2, 5, 10, 30, 60, 120),
        )
        self.error_counter = Counter(
            "processing_errors_total", "Total processing errors"
        )
        self.queue_size = Gauge(
            "processing_queue_size", "Current size of processing queue"
        )
        self.gpu_memory = Gauge("gpu_memory_usage_bytes", "GPU memory usage in bytes")


class EnhancedTagGenerator:
    """Advanced tag generation with multiple models."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_models()
        self.setup_categories()

    def setup_models(self):
        """Initialize models for tag generation."""
        # CLIP for zero-shot classification
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)

        # BLIP for image understanding
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model.to(self.device)

    def setup_categories(self):
        """Setup jewelry-specific categories and attributes."""
        self.jewelry_categories = [
            "ring",
            "necklace",
            "bracelet",
            "earrings",
            "pendant",
            "chain",
            "gemstone",
            "diamond",
            "gold",
            "silver",
        ]

        self.jewelry_attributes = [
            "vintage",
            "modern",
            "elegant",
            "casual",
            "ornate",
            "minimalist",
            "luxury",
            "handmade",
            "antique",
            "designer",
        ]

        self.material_descriptors = [
            "gold",
            "silver",
            "platinum",
            "rose gold",
            "white gold",
            "sterling silver",
            "brass",
            "copper",
            "titanium",
        ]

    async def generate_tags(self, image: Image.Image) -> Dict:
        """Generate comprehensive tags using multiple models."""
        # Get CLIP-based category predictions
        category_scores = await self._get_clip_predictions(
            image, self.jewelry_categories
        )

        # Get attribute predictions
        attribute_scores = await self._get_clip_predictions(
            image, self.jewelry_attributes
        )

        # Get material predictions
        material_scores = await self._get_clip_predictions(
            image, self.material_descriptors
        )

        # Get BLIP caption for additional context
        caption = await self.generate_caption(image)

        # Combine and filter results
        categories = [cat for cat, score in category_scores.items() if score > 0.3]

        attributes = [attr for attr, score in attribute_scores.items() if score > 0.3]

        descriptors = [desc for desc, score in material_scores.items() if score > 0.3]

        # Add caption-based tags
        caption_tags = await self.extract_tags_from_caption(caption)
        descriptors.extend(caption_tags)

        # Calculate confidence scores
        confidence = [score for score in category_scores.values() if score > 0.3]

        return {
            "categories": categories,
            "attributes": attributes,
            "descriptors": list(set(descriptors)),
            "confidence": confidence,
        }

    async def _get_clip_predictions(
        self, image: Image.Image, candidates: List[str]
    ) -> Dict[str, float]:
        """Get CLIP-based predictions for candidates."""
        # Prepare image and text inputs
        image_inputs = self.clip_processor(images=image, return_tensors="pt").to(
            self.device
        )

        text_inputs = self.clip_processor(
            text=candidates, return_tensors="pt", padding=True
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**image_inputs)
            text_features = self.clip_model.get_text_features(**text_inputs)

            # Normalize features
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

            # Calculate similarity scores
            similarity = (image_features @ text_features.T).squeeze(0)

        # Convert to dictionary
        scores = {
            candidate: score.item() for candidate, score in zip(candidates, similarity)
        }

        return scores

    async def generate_caption(self, image: Image.Image) -> str:
        """Generate image caption using BLIP."""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.blip_model.generate(**inputs)
            caption = self.blip_processor.decode(
                generated_ids[0], skip_special_tokens=True
            )

        return caption

    async def extract_tags_from_caption(self, caption: str) -> List[str]:
        """Extract relevant tags from caption."""
        # Basic tokenization and filtering
        words = caption.lower().split()

        # Remove common words and keep relevant terms
        stopwords = {"a", "an", "the", "is", "are", "with", "of", "and"}
        tags = [word for word in words if word not in stopwords and len(word) > 2]

        return list(set(tags))


class JewelryProcessor:
    """Advanced jewelry image processor with distributed processing capabilities."""

    def __init__(self, config: Dict):
        self.config = config
        self.metrics = ProcessingMetrics()
        self.detector = JewelryDetector(config["detector"])
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 4))
        self.setup_connections()
        self.tag_generator = EnhancedTagGenerator()
        self.logger = logging.getLogger(__name__)

    def setup_connections(self):
        """Initialize connections to external services."""
        self.redis = aioredis.from_url(self.config["redis_url"])
        self.mongo = AsyncIOMotorClient(self.config["mongo_url"])
        self.db = self.mongo[self.config["database_name"]]

        # Setup Cloudinary
        cloudinary.config(
            cloud_name=self.config["cloudinary"]["cloud_name"],
            api_key=self.config["cloudinary"]["api_key"],
            api_secret=self.config["cloudinary"]["api_secret"],
        )

    async def process_queue(self):
        """Main processing loop for queue items."""
        while True:
            try:
                # Update queue size metric
                queue_size = await self.redis.llen("image_queue")
                self.metrics.queue_size.set(queue_size)

                # Get item from queue
                item = await self.redis.blpop("image_queue", timeout=1)
                if not item:
                    await asyncio.sleep(1)
                    continue

                # Process item
                _, image_data = item
                job_info = json.loads(image_data)

                result = await self.process_image(
                    job_info["image_path"], job_info.get("options", {})
                )

                # Store results
                await self.store_results(result)

                # Update progress
                await self.update_job_progress(job_info["job_id"], result)

            except Exception as e:
                self.metrics.error_counter.inc()
                self.logger.error(
                    f"Processing error: {
                        traceback.format_exc()}"
                )
                await self.handle_processing_error(e)

    async def process_image(
        self, image_path: Union[str, Path], options: Dict
    ) -> ProcessingResult:
        """Process a single image with extensive error handling."""
        start_time = time.time()

        try:
            # Load and preprocess image
            image = await self.load_image(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Perform detection
            with self.metrics.processing_time.time():
                detections = await self.run_detection(image)

            # Apply enhancements and processing
            processed_image = await self.apply_processing(image, detections, options)

            # Upload to Cloudinary
            upload_result = await self.upload_to_cloudinary(processed_image)

            # Generate metadata
            metadata = await self.generate_metadata(
                image_path, detections, options, upload_result
            )

            processing_time = time.time() - start_time

            return ProcessingResult(
                image_id=str(Path(image_path).stem),
                detections=detections,
                metadata=metadata,
                processed_url=upload_result["secure_url"],
                processing_time=processing_time,
            )

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            self.metrics.error_counter.inc()
            raise

    async def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load and validate image."""
        try:
            if isinstance(image_path, str) and image_path.startswith("http"):
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_path) as response:
                        if response.status != 200:
                            raise ValueError(
                                f"Failed to fetch image from URL: {image_path}"
                            )
                        image_data = await response.read()
                        image = Image.open(io.BytesIO(image_data)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")

            # Convert to numpy array
            image_array = np.array(image)

            # Validate image
            if image_array.size == 0:
                raise ValueError("Empty image")

            return image_array

        except Exception as e:
            self.logger.error(
                f"Image loading failed for {image_path}: {
                    str(e)}"
            )
            raise

    async def run_detection(self, image: np.ndarray) -> DetectionResult:
        """Run detection in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.detector.detect, image)

    async def apply_processing(
        self, image: np.ndarray, detections: DetectionResult, options: Dict
    ) -> Image.Image:
        """Apply post-processing based on detections and options."""
        processed = image.copy()

        # Initialize EnhancedJewelryProcessor for advanced processing
        enhanced_processor = EnhancedJewelryProcessor(
            self.config, self.tag_generator, self.executor, self.metrics
        )

        # Apply enhancements
        if options.get("enhance", False):
            processed = await enhanced_processor.enhance_image(processed, options)

        # Draw detections if requested
        if options.get("draw_detections", False):
            processed = enhanced_processor.draw_detections(processed, detections)

        return Image.fromarray(processed)

    async def upload_to_cloudinary(self, image: Image.Image) -> Dict:
        """Upload processed image to Cloudinary."""
        loop = asyncio.get_running_loop()

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        return await loop.run_in_executor(
            self.executor, cloudinary.uploader.upload, buffer
        )

    async def generate_metadata(
        self,
        image_path: Union[str, Path],
        detections: DetectionResult,
        options: Dict,
        upload_result: Dict,
    ) -> Dict:
        """Generate comprehensive metadata."""
        metadata = {
            "original_path": str(image_path),
            "processing_options": options,
            "detections": {
                "count": len(detections.boxes),
                "labels": detections.labels,
                "scores": detections.scores.tolist(),
                "boxes": detections.boxes.tolist(),
            },
            "cloudinary": {
                "public_id": upload_result.get("public_id"),
                "url": upload_result.get("secure_url"),
                "version": upload_result.get("version"),
            },
            "timestamp": time.time(),
        }

        if self.config.get("generate_embeddings", False):
            embeddings = await self.generate_embeddings(Image.open(image_path))
            metadata["embeddings"] = embeddings.tolist()

        return metadata

    async def generate_embeddings(self, image: Image.Image) -> np.ndarray:
        """Generate image embeddings using CLIP."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, self._generate_embeddings_sync, image
        )

    def _generate_embeddings_sync(self, image: Image.Image) -> np.ndarray:
        """Synchronous method to generate embeddings."""
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model.to(device)

        inputs = clip_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)

        return image_features.cpu().numpy()

    async def store_results(self, result: ProcessingResult):
        """Store processing results in MongoDB."""
        await self.db.results.insert_one(
            {
                "image_id": result.image_id,
                "detections": {
                    "boxes": result.detections.boxes.tolist(),
                    "labels": result.detections.labels,
                    "scores": result.detections.scores.tolist(),
                },
                "metadata": result.metadata,
                "processed_url": result.processed_url,
                "processing_time": result.processing_time,
            }
        )

    async def update_job_progress(self, job_id: str, result: ProcessingResult):
        """Update job progress in Redis."""
        await self.redis.hset(
            f"job:{job_id}",
            mapping={
                "status": "completed",
                "result": json.dumps(
                    {"image_id": result.image_id, "processed_url": result.processed_url}
                ),
            },
        )

    async def handle_processing_error(self, error: Exception):
        """Handle processing errors with retries."""
        self.logger.error(f"Processing error: {str(error)}")
        # Implement error handling logic here
        # For example, move to error queue, retry, etc.
        # Placeholder for error handling
        pass

    async def process_all(self):
        """Process all images in batches."""
        all_results = []
        batches = self.get_image_batches()

        for batch in batches:
            results = await self.process_batch(batch)
            all_results.extend(results)

        return all_results

    def get_image_batches(self) -> List[List[Path]]:
        """Retrieve image batches based on the configuration."""
        input_paths = self.config.input_paths
        batch_size = self.config.batch_size
        return [
            input_paths[i : i + batch_size]
            for i in range(0, len(input_paths), batch_size)
        ]

    async def process_batch(self, batch: List[Path]) -> List[Dict]:
        """Process a batch of images with metadata extraction."""
        results = []

        tasks = [self.process_image(path, self.config.__dict__) for path in batch]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                self.metrics.error_counter.inc()
                logger.error(
                    f"Batch processing error: {
                        traceback.format_exc()}"
                )
                continue
            results.append(result)

        return results


class EnhancedJewelryProcessor(JewelryProcessor):
    """Enhanced processor with advanced pipeline capabilities."""

    def __init__(
        self,
        config: Dict,
        tag_generator: EnhancedTagGenerator,
        executor: ThreadPoolExecutor,
        metrics: ProcessingMetrics,
    ):
        super().__init__(config)
        self.tag_generator = tag_generator
        self.executor = executor
        self.metrics = metrics
        self.setup_enhanced_pipeline()

    def setup_enhanced_pipeline(self):
        """Initialize enhanced processing pipeline."""
        self.augmentation = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), p=0.5),
                A.HueSaturationValue(p=0.3),
                A.CLAHE(p=0.5),
                A.AdvancedBlur(blur_limit=(3, 7), p=0.3),
            ]
        )

        self.enhancement_pipeline = {
            "color_enhancement": self.enhance_colors,
            "detail_enhancement": self.enhance_details,
            "noise_reduction": self.reduce_noise,
            "background_enhancement": self.enhance_background,
            "highlight_recovery": self.recover_highlights,
            "shadow_recovery": self.recover_shadows,
        }

    async def process_batch(
        self, image_paths: List[Union[str, Path]], options: Dict
    ) -> List[ProcessingResult]:
        """Process a batch of images concurrently."""
        batch_size = options.get("batch_size", self.config.batch_size)
        results = []

        # Split into batches
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]

            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[self.process_image(path, options) for path in batch],
                return_exceptions=True,
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    self.metrics.error_counter.inc()
                    logger.error(
                        f"Error processing image in batch: {
                            traceback.format_exc()}"
                    )
                    continue
                results.append(result)

            # Update metrics
            self.metrics.queue_size.dec(len(batch))

        return results

    async def enhance_image(self, image: np.ndarray, options: Dict) -> np.ndarray:
        """Enhanced image processing pipeline."""
        try:
            # Apply augmentations if specified
            if options.get("augment", False):
                augmented = self.augmentation(image=image)
                image = augmented["image"]

            # Apply selected enhancements
            for enhancement, params in options.get("enhancements", {}).items():
                if enhancement in self.enhancement_pipeline and params.get(
                    "enabled", False
                ):
                    image = await self.enhancement_pipeline[enhancement](
                        image, params.get("strength", 0.5)
                    )

            # Apply custom processing steps
            if options.get("custom_pipeline"):
                image = await self.apply_custom_pipeline(
                    image, options["custom_pipeline"]
                )

            return image

        except Exception as e:
            self.logger.error(f"Enhancement failed: {str(e)}")
            raise

    async def enhance_colors(
        self, image: np.ndarray, strength: float = 0.5
    ) -> np.ndarray:
        """Enhance image colors."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0 * strength, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Enhance AB channels
        a = cv2.addWeighted(a, 1 + strength, a, 0, 0)
        b = cv2.addWeighted(b, 1 + strength, b, 0, 0)

        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    async def enhance_details(
        self, image: np.ndarray, strength: float = 0.5
    ) -> np.ndarray:
        """Enhance image details using frequency separation."""
        # Convert to float
        img_float = image.astype(np.float32) / 255.0

        # Create gaussian blur
        blur = cv2.GaussianBlur(img_float, (0, 0), 3)
        high_freq = img_float - blur

        # Enhance high frequency details
        enhanced = img_float + high_freq * strength

        # Clip and convert back
        return np.clip(enhanced * 255, 0, 255).astype(np.uint8)

    async def recover_highlights(
        self, image: np.ndarray, strength: float = 0.5
    ) -> np.ndarray:
        """Recover blown out highlights."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # Create mask for highlights
        highlight_mask = v > 230

        # Reduce brightness in highlight areas
        v[highlight_mask] = v[highlight_mask] * (1 - strength)

        # Merge and convert back
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    async def recover_shadows(
        self, image: np.ndarray, strength: float = 0.5
    ) -> np.ndarray:
        """Recover shadow details."""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Create mask for shadows
        shadow_mask = l < 50

        # Enhance brightness in shadow areas
        l[shadow_mask] = l[shadow_mask] * (1 + strength)

        # Merge and convert back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    async def reduce_noise(
        self, image: np.ndarray, strength: float = 0.5
    ) -> np.ndarray:
        """Reduce noise in the image."""
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=int(10 * strength),
            hColor=int(10 * strength),
            templateWindowSize=7,
            searchWindowSize=21,
        )

    async def enhance_background(
        self, image: np.ndarray, strength: float = 0.5
    ) -> np.ndarray:
        """Enhance the background of the image."""
        # Placeholder for background enhancement logic
        # Example: Apply Gaussian blur to the background
        mask = self.create_background_mask(image)
        blurred = cv2.GaussianBlur(image, (21, 21), 0)
        enhanced = np.where(mask[:, :, None], blurred, image)
        return enhanced

    def create_background_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a mask for the background."""
        # Simple threshold based on color or other criteria
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return mask.astype(bool)

    async def apply_custom_pipeline(
        self, image: np.ndarray, pipeline: List[Dict]
    ) -> np.ndarray:
        """Apply custom processing pipeline."""
        for step in pipeline:
            if step["type"] == "filter":
                image = await self.apply_filter(image, step)
            elif step["type"] == "adjustment":
                image = await self.apply_adjustment(image, step)
            elif step["type"] == "transform":
                image = await self.apply_transform(image, step)
        return image

    async def apply_filter(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply image filter."""
        filter_type = params.get("filter_type")
        strength = params.get("strength", 1.0)

        if filter_type == "bilateral":
            return cv2.bilateralFilter(
                image, d=9, sigmaColor=75 * strength, sigmaSpace=75 * strength
            )
        elif filter_type == "guided":
            return cv2.ximgproc.guidedFilter(image, image, radius=8, eps=100 * strength)
        else:
            return image

    async def apply_adjustment(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply image adjustment."""
        adjustment_type = params.get("adjustment_type")
        value = params.get("value", 1.0)

        if adjustment_type == "exposure":
            return cv2.convertScaleAbs(image, alpha=value, beta=0)
        elif adjustment_type == "contrast":
            return cv2.convertScaleAbs(image, alpha=value, beta=128 * (1 - value))

        return image

    async def apply_transform(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply image transform."""
        transform_type = params.get("transform_type")

        if transform_type == "rotate":
            angle = params.get("angle", 0)
            return self._rotate_image(image, angle)
        elif transform_type == "crop":
            bbox = params.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                return image[y1:y2, x1:x2]
        return image

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by a given angle."""
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
        return rotated

    def draw_detections(
        self, image: np.ndarray, detections: DetectionResult
    ) -> np.ndarray:
        """Draw detection boxes and labels."""
        for box, label, score in zip(
            detections.boxes, detections.labels, detections.scores
        ):
            x1, y1, x2, y2 = box.astype(int)

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            text = f"{label}: {score:.2f}"
            cv2.putText(
                image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return image


async def main():
    # Load config
    with open("config/config.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = BatchProcessingConfig(
        input_paths=[Path(p) for p in config_dict.get("input_paths", [])],
        output_dir=Path(config_dict["output_dir"]),
        batch_size=config_dict.get("batch_size", 32),
        generate_embeddings=config_dict.get("generate_embeddings", True),
        extract_metadata=config_dict.get("extract_metadata", True),
        detect_objects=config_dict.get("detect_objects", True),
        generate_tags=config_dict.get("generate_tags", True),
        export_formats=config_dict.get("export_formats", ["csv", "json", "parquet"]),
        cloudinary_config=config_dict.get("cloudinary_config"),
    )

    # Initialize processor
    processor = JewelryProcessor(config.__dict__)

    # Optionally initialize Ray if enabled
    if config.use_ray:
        ray_init()

    # Start processing queue
    await processor.process_queue()


if __name__ == "__main__":
    asyncio.run(main())
