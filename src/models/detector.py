# src/models/detector.py
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)
@dataclass
class DetectionResult:
    """Container for detection results."""
    boxes: List[List[int]]
    labels: List[str]
    scores: List[float]
    
@dataclass
class TagResult:
    """Container for tag results."""
    categories: List[str]
    attributes: List[str]
    descriptors: List[str]
    confidence: List[float]

class EnhancedObjectDetector:
    """Advanced object detection with multiple model support."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_models()
        
    def setup_models(self):
        """Initialize detection models."""
        # DETR for object detection
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.detr_model.to(self.device)
        
        # CLIP for visual understanding
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        
        # BLIP for image captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model.to(self.device)
        
    async def detect_objects(self, image: Image.Image) -> DetectionResult:
        """Detect objects using DETR."""
        # Prepare image
        inputs = self.detr_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.detr_model(**inputs)
            
        # Process results
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7
        
        # Convert boxes to image coordinates
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        postprocessed_outputs = self.detr_processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=0.7
        )[0]
        
        boxes = postprocessed_outputs['boxes'].cpu().numpy().tolist()
        labels = [
            self.detr_model.config.id2label[label.item()]
            for label in postprocessed_outputs['labels']
        ]
        scores = postprocessed_outputs['scores'].cpu().numpy().tolist()
        
        return DetectionResult(boxes=boxes, labels=labels, scores=scores)

@dataclass
class DetectionResult:
    """Container for detection results."""
    boxes: np.ndarray
    labels: List[str]
    scores: np.ndarray
    embeddings: np.ndarray

class JewelryDetector:
    """Advanced jewelry detection model with feature extraction."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing detector on device: {self.device}")
        self.model = self._build_model()
        self.preprocessing = ImagePreprocessor(config['preprocessing'])
        
    def _build_model(self) -> nn.Module:
        """Build and configure detection model."""
        logger.debug("Building detection model")
        model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        
        # Modify for jewelry detection
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        num_classes = len(self.config['classes']) + 1  # +1 for background
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        model.to(self.device)
        model.eval()
        return model
    
    @torch.no_grad()
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Perform jewelry detection with feature extraction."""
        try:
            # Preprocess image
            processed_image = self.preprocessing(image)
            tensor_image = torch.from_numpy(processed_image).permute(2, 0, 1).float() / 255.0
            tensor_image = tensor_image.unsqueeze(0).to(self.device)
            
            # Get predictions
            predictions = self.model(tensor_image)[0]
            
            # Extract high confidence predictions
            mask = predictions['scores'] > self.config['confidence_threshold']
            boxes = predictions['boxes'][mask].cpu().numpy()
            scores = predictions['scores'][mask].cpu().numpy()
            labels = [
                self.config['classes'][i]
                for i in predictions['labels'][mask].cpu().numpy()
            ]
            
            # Extract feature embeddings
            embeddings = self._extract_embeddings(tensor_image, boxes)
            
            return DetectionResult(
                boxes=boxes,
                labels=labels,
                scores=scores,
                embeddings=embeddings
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            raise
    
    def _extract_embeddings(self, image: torch.Tensor, boxes: np.ndarray) -> np.ndarray:
        """Extract feature embeddings for detected regions."""
        features = self.model.backbone(image)
        embeddings = []
        
        for box in boxes:
            # Extract ROI features
            roi = self._extract_roi_features(features, box)
            embeddings.append(roi.cpu().numpy())
            
        return np.array(embeddings)
    
    def _extract_roi_features(self, features: Dict[str, torch.Tensor], box: np.ndarray) -> torch.Tensor:
        """Extract ROI features from backbone features."""
        # Get highest resolution feature map
        feature_map = features['0']
        
        # Scale box coordinates to feature map size
        scaled_box = self._scale_box_to_feature_map(box, feature_map.shape[-2:])
        
        # Extract and pool features
        roi_features = self.model.roi_heads.box_roi_pool(
            feature_map,
            [torch.tensor([scaled_box]).to(self.device)],
            [feature_map.shape[-2:]]
        )
        
        return roi_features.mean([2, 3]).squeeze(0)
    
    @staticmethod
    def _scale_box_to_feature_map(box: np.ndarray, feature_size: Tuple[int, int]) -> np.ndarray:
        """Scale box coordinates to feature map dimensions."""
        scale_x = feature_size[1] / box[2]
        scale_y = feature_size[0] / box[3]
        
        return np.array([
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y
        ])

class ImagePreprocessor:
    """Advanced image preprocessing pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline."""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Apply preprocessing steps
            if self.config.get('denoise', False):
                image = cv2.fastNlMeansDenoisingColored(
                    image,
                    None,
                    h=self.config['denoise_strength'],
                    searchWindowSize=21,
                    blockSize=7
                )
            
            if self.config.get('sharpen', False):
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                image = cv2.filter2D(image, -1, kernel)
            
            if self.config.get('contrast_boost', False):
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l_channel = lab[:,:,0]
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l_channel)
                lab[:,:,0] = cl
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    config = {
        'classes': ['ring', 'necklace', 'bracelet', 'earring', 'pendant'],
        'confidence_threshold': 0.7,
        'preprocessing': {
            'denoise': True,
            'denoise_strength': 10,
            'sharpen': True,
            'contrast_boost': True
        }
    }
    
    detector = JewelryDetector(config)
    
    # Test detection
    image = cv2.imread('test_image.jpg')
    result = detector.detect(image)
    print(f"Detected {len(result.boxes)} objects")
    print(result.labels)
    print(result.scores)
    print(result.boxes)
    print(result.embeddings)