"""Main HOI detector implementation."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from pathlib import Path

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("Warning: Detectron2 not available. Install with: pip install detectron2")

from .interaction_head import InteractionHead
from .relationship_detector import RelationshipDetector


class HOIDetector:
    """Main Human-Object Interaction detector.
    
    This class provides a unified interface for HOI detection using
    various backends (Detectron2, custom models, etc.).
    
    Args:
        config_path: Path to configuration file
        device: Device to run inference on ('cuda', 'mps', 'cpu')
        score_threshold: Minimum confidence score for detections
        backend: Backend to use ('detectron2', 'custom')
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: str = "auto",
        score_threshold: float = 0.5,
        backend: str = "detectron2"
    ):
        self.device = self._setup_device(device)
        self.score_threshold = score_threshold
        self.backend = backend
        
        if backend == "detectron2" and DETECTRON2_AVAILABLE:
            self._setup_detectron2()
        elif backend == "custom":
            self._setup_custom_model()
        else:
            raise ValueError(f"Backend '{backend}' not supported or not available")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup device with automatic fallback."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _setup_detectron2(self) -> None:
        """Setup Detectron2 predictor."""
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
        self.cfg.MODEL.DEVICE = str(self.device)
        
        self.predictor = DefaultPredictor(self.cfg)
        
        # Initialize interaction head for HOI detection
        self.interaction_head = InteractionHead(
            num_classes=80,
            num_interactions=600,  # HICO-DET has 600 interactions
            hidden_dim=256
        ).to(self.device)
        
        # Initialize relationship detector
        self.relationship_detector = RelationshipDetector(
            feature_dim=256,
            num_relationships=600
        ).to(self.device)
    
    def _setup_custom_model(self) -> None:
        """Setup custom model architecture."""
        # This would be implemented for custom architectures
        raise NotImplementedError("Custom model backend not yet implemented")
    
    def predict(
        self, 
        image: Union[str, Path, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Predict HOI detections on an image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Dictionary containing:
                - 'instances': Object detection results
                - 'interactions': HOI interaction predictions
                - 'relationships': Relationship scores
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        # Convert BGR to RGB for Detectron2
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image
        
        # Get object detections
        outputs = self.predictor(image_rgb)
        
        # Extract features for HOI detection
        if hasattr(outputs, 'instances') and len(outputs.instances) > 0:
            # Get detected objects
            instances = outputs.instances
            
            # Filter by score threshold
            valid_mask = instances.scores >= self.score_threshold
            if valid_mask.sum() > 0:
                instances = instances[valid_mask]
                
                # Detect interactions
                interactions = self._detect_interactions(image_rgb, instances)
                
                # Detect relationships
                relationships = self._detect_relationships(image_rgb, instances)
                
                return {
                    'instances': instances,
                    'interactions': interactions,
                    'relationships': relationships
                }
        
        return {
            'instances': None,
            'interactions': None,
            'relationships': None
        }
    
    def _detect_interactions(
        self, 
        image: np.ndarray, 
        instances
    ) -> torch.Tensor:
        """Detect interactions between detected objects."""
        # This is a simplified implementation
        # In practice, you would extract features and use the interaction head
        num_objects = len(instances)
        num_interactions = 600  # HICO-DET interactions
        
        # Create interaction matrix (simplified)
        interactions = torch.zeros((num_objects, num_objects, num_interactions))
        
        # For now, return random interactions (placeholder)
        interactions = torch.randn((num_objects, num_objects, num_interactions))
        
        return interactions
    
    def _detect_relationships(
        self, 
        image: np.ndarray, 
        instances
    ) -> torch.Tensor:
        """Detect relationships between objects."""
        num_objects = len(instances)
        num_relationships = 600
        
        # Create relationship matrix (simplified)
        relationships = torch.zeros((num_objects, num_objects, num_relationships))
        
        # For now, return random relationships (placeholder)
        relationships = torch.randn((num_objects, num_objects, num_relationships))
        
        return relationships
    
    def visualize(
        self, 
        image: Union[str, Path, np.ndarray],
        results: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Visualize HOI detection results.
        
        Args:
            image: Input image
            results: Detection results from predict()
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image as numpy array
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        # Convert BGR to RGB for visualization
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image
        
        # Create visualizer
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        v = Visualizer(image_rgb, metadata, scale=1.2)
        
        # Draw object detections
        if results['instances'] is not None:
            v = v.draw_instance_predictions(results['instances'].to("cpu"))
        
        # Get visualization
        vis_image = v.get_image()[:, :, ::-1]  # Convert back to BGR
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def batch_predict(
        self, 
        images: List[Union[str, Path, np.ndarray]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Predict HOI detections on a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of detection results for each image
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
