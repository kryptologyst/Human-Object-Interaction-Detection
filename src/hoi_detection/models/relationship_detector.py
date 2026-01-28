"""Relationship detector for HOI detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RelationshipDetector(nn.Module):
    """Relationship detector for identifying object relationships.
    
    This module detects spatial and semantic relationships between
    objects in the scene.
    
    Args:
        feature_dim: Input feature dimension
        num_relationships: Number of relationship types
        hidden_dim: Hidden dimension size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_relationships: int = 600,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_relationships = num_relationships
        self.hidden_dim = hidden_dim
        
        # Spatial relationship encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim // 2),  # 8 spatial features
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Semantic relationship encoder
        self.semantic_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Relationship classification head
        self.relationship_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relationships)
        )
        
        # Attention mechanism for relationship focus
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for relationship detection.
        
        Args:
            features: Object features [N, feature_dim]
            boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
            labels: Object labels [N]
            
        Returns:
            Relationship scores [N, N, num_relationships]
        """
        num_objects = features.shape[0]
        
        # Compute spatial features
        spatial_features = self._compute_spatial_features(boxes)
        
        # Compute semantic features
        semantic_features = self._compute_semantic_features(features, labels)
        
        # Combine spatial and semantic features
        combined_features = torch.cat([semantic_features, spatial_features], dim=-1)
        
        # Apply attention
        attended_features, _ = self.attention(
            combined_features, combined_features, combined_features
        )
        
        # Predict relationships
        relationship_scores = self.relationship_head(attended_features)
        
        # Reshape to [N, N, num_relationships]
        relationship_scores = relationship_scores.view(
            num_objects, num_objects, self.num_relationships
        )
        
        return relationship_scores
    
    def _compute_spatial_features(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute spatial relationship features.
        
        Args:
            boxes: Bounding boxes [N, 4]
            
        Returns:
            Spatial features [N, N, hidden_dim//2]
        """
        num_objects = boxes.shape[0]
        
        # Compute pairwise spatial features
        spatial_features = []
        
        for i in range(num_objects):
            for j in range(num_objects):
                if i == j:
                    # Self-relationship
                    spatial_feat = torch.zeros(8)
                else:
                    spatial_feat = self._compute_pairwise_spatial_features(
                        boxes[i], boxes[j]
                    )
                spatial_features.append(spatial_feat)
        
        spatial_features = torch.stack(spatial_features).view(
            num_objects, num_objects, 8
        )
        
        # Encode spatial features
        spatial_features = spatial_features.view(-1, 8)
        spatial_features = self.spatial_encoder(spatial_features)
        spatial_features = spatial_features.view(
            num_objects, num_objects, self.hidden_dim // 2
        )
        
        return spatial_features
    
    def _compute_pairwise_spatial_features(
        self, 
        box1: torch.Tensor, 
        box2: torch.Tensor
    ) -> torch.Tensor:
        """Compute spatial features between two boxes.
        
        Args:
            box1: First bounding box [4]
            box2: Second bounding box [4]
            
        Returns:
            Spatial features [8]
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Compute centers
        cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        # Compute sizes
        w1, h1 = x2_1 - x1_1, y2_1 - y1_1
        w2, h2 = x2_2 - x1_2, y2_2 - y1_2
        
        # Compute spatial relationships
        dx = cx2 - cx1
        dy = cy2 - cy1
        distance = torch.sqrt(dx**2 + dy**2)
        angle = torch.atan2(dy, dx)
        
        # Size ratios
        size_ratio_w = w2 / (w1 + 1e-8)
        size_ratio_h = h2 / (h1 + 1e-8)
        
        # IoU
        iou = self._compute_iou(box1, box2)
        
        spatial_features = torch.stack([
            dx, dy, distance, angle,
            size_ratio_w, size_ratio_h, iou, torch.tensor(0.0)  # placeholder
        ])
        
        return spatial_features
    
    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Compute Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Compute intersection
        x1_i = torch.max(x1_1, x1_2)
        y1_i = torch.max(y1_1, y1_2)
        x2_i = torch.min(x2_1, x2_2)
        y2_i = torch.min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return torch.tensor(0.0)
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)
    
    def _compute_semantic_features(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute semantic relationship features.
        
        Args:
            features: Object features [N, feature_dim]
            labels: Object labels [N]
            
        Returns:
            Semantic features [N, N, hidden_dim]
        """
        num_objects = features.shape[0]
        
        # Create pairwise semantic features
        features_i = features.unsqueeze(1).expand(-1, num_objects, -1)
        features_j = features.unsqueeze(0).expand(num_objects, -1, -1)
        
        # Concatenate pairwise features
        pairwise_features = torch.cat([features_i, features_j], dim=-1)
        
        # Encode semantic features
        semantic_features = pairwise_features.view(-1, self.feature_dim * 2)
        semantic_features = self.semantic_encoder(semantic_features)
        semantic_features = semantic_features.view(
            num_objects, num_objects, self.hidden_dim
        )
        
        return semantic_features
    
    def compute_relationship_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute relationship detection loss.
        
        Args:
            predictions: Predicted relationship scores [N, N, num_relationships]
            targets: Ground truth relationship labels [N, N, num_relationships]
            weights: Optional loss weights
            
        Returns:
            Relationship loss
        """
        # Flatten predictions and targets
        pred_flat = predictions.view(-1, self.num_relationships)
        target_flat = targets.view(-1, self.num_relationships)
        
        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(
            pred_flat, target_flat.float(), weight=weights
        )
        
        return loss
