"""Human-Object Interaction Detection Package.

This package provides state-of-the-art models and tools for detecting
human-object interactions in images and videos.
"""

__version__ = "1.0.0"
__author__ = "AI Projects"

from .models import HOIDetector, InteractionHead, RelationshipDetector
from .data import HOIDataset, HICODataset, VCOCODataset
from .utils import visualize_hoi, compute_hoi_metrics, setup_device
from .train import HOITrainer
from .eval import HOIEvaluator

__all__ = [
    "HOIDetector",
    "InteractionHead", 
    "RelationshipDetector",
    "HOIDataset",
    "HICODataset",
    "VCOCODataset",
    "visualize_hoi",
    "compute_hoi_metrics",
    "setup_device",
    "HOITrainer",
    "HOIEvaluator",
]
