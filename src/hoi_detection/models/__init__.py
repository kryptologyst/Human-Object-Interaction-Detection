"""Core HOI detection models and architectures."""

from .detector import HOIDetector
from .interaction_head import InteractionHead
from .relationship_detector import RelationshipDetector
from .baseline import BaselineHOIDetector
from .transformer import TransformerHOIDetector

__all__ = [
    "HOIDetector",
    "InteractionHead",
    "RelationshipDetector", 
    "BaselineHOIDetector",
    "TransformerHOIDetector",
]
