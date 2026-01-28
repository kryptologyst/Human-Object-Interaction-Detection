"""Interaction head for HOI detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class InteractionHead(nn.Module):
    """Interaction head for detecting human-object interactions.
    
    This module takes object features and predicts interaction scores
    between detected objects.
    
    Args:
        num_classes: Number of object classes
        num_interactions: Number of interaction types
        hidden_dim: Hidden dimension size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        num_interactions: int = 600,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_interactions = num_interactions
        self.hidden_dim = hidden_dim
        
        # Feature projection layers
        self.feature_proj = nn.Sequential(
            nn.Linear(2048, hidden_dim),  # Assuming 2048-dim features from backbone
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Interaction classification head
        self.interaction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Pairwise features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_interactions)
        )
        
        # Spatial attention for interaction regions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for interaction detection.
        
        Args:
            features: Object features [N, C, H, W]
            boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
            labels: Object labels [N]
            
        Returns:
            Interaction scores [N, N, num_interactions]
        """
        batch_size, num_objects = boxes.shape[0], boxes.shape[0]
        
        # Project features
        projected_features = self.feature_proj(features.view(batch_size, -1))
        
        # Create pairwise features
        pairwise_features = self._create_pairwise_features(
            projected_features, boxes, labels
        )
        
        # Predict interactions
        interaction_scores = self.interaction_head(pairwise_features)
        
        # Reshape to [N, N, num_interactions]
        interaction_scores = interaction_scores.view(
            num_objects, num_objects, self.num_interactions
        )
        
        return interaction_scores
    
    def _create_pairwise_features(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Create pairwise features for interaction detection.
        
        Args:
            features: Object features [N, hidden_dim]
            boxes: Bounding boxes [N, 4]
            labels: Object labels [N]
            
        Returns:
            Pairwise features [N*N, hidden_dim*2]
        """
        num_objects = features.shape[0]
        
        # Expand features for all pairs
        features_i = features.unsqueeze(1).expand(-1, num_objects, -1)
        features_j = features.unsqueeze(0).expand(num_objects, -1, -1)
        
        # Concatenate pairwise features
        pairwise_features = torch.cat([features_i, features_j], dim=-1)
        
        # Flatten to [N*N, hidden_dim*2]
        pairwise_features = pairwise_features.view(-1, features.shape[1] * 2)
        
        return pairwise_features
    
    def compute_interaction_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute interaction classification loss.
        
        Args:
            predictions: Predicted interaction scores [N, N, num_interactions]
            targets: Ground truth interaction labels [N, N, num_interactions]
            weights: Optional loss weights
            
        Returns:
            Interaction loss
        """
        # Flatten predictions and targets
        pred_flat = predictions.view(-1, self.num_interactions)
        target_flat = targets.view(-1, self.num_interactions)
        
        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(
            pred_flat, target_flat.float(), weight=weights
        )
        
        return loss
