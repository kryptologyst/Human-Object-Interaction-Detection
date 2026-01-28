"""Utility functions for HOI detection."""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


def setup_device(device: str = "auto") -> torch.device:
    """Setup device with automatic fallback.
    
    Args:
        device: Device preference ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        Available torch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_hoi(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: List[str],
    interactions: Optional[np.ndarray] = None,
    interaction_classes: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> np.ndarray:
    """Visualize HOI detection results.
    
    Args:
        image: Input image [H, W, C]
        boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
        labels: Object labels [N]
        interactions: Interaction matrix [N, N, num_interactions]
        interaction_classes: List of interaction class names
        save_path: Optional path to save visualization
        show: Whether to display the image
        
    Returns:
        Visualization image
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw bounding boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(boxes)))
    
    for i, (box, label, color) in enumerate(zip(boxes, labels, colors)):
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(x1, y1 - 5, label, fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Draw interactions
    if interactions is not None and interaction_classes is not None:
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i != j and interactions[i, j].sum() > 0:
                    # Find the strongest interaction
                    interaction_idx = interactions[i, j].argmax()
                    interaction_name = interaction_classes[interaction_idx]
                    
                    # Draw interaction arrow
                    box1_center = [(boxes[i][0] + boxes[i][2]) / 2, 
                                  (boxes[i][1] + boxes[i][3]) / 2]
                    box2_center = [(boxes[j][0] + boxes[j][2]) / 2, 
                                  (boxes[j][1] + boxes[j][3]) / 2]
                    
                    ax.annotate('', xy=box2_center, xytext=box1_center,
                              arrowprops=dict(arrowstyle='->', color='red', lw=2))
                    
                    # Add interaction label
                    mid_point = [(box1_center[0] + box2_center[0]) / 2,
                                (box1_center[1] + box2_center[1]) / 2]
                    ax.text(mid_point[0], mid_point[1], interaction_name,
                           fontsize=8, color='red', ha='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def compute_hoi_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute HOI detection metrics.
    
    Args:
        predictions: Predicted interaction scores [N, N, num_interactions]
        targets: Ground truth interaction labels [N, N, num_interactions]
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy for easier computation
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    # Apply threshold
    pred_binary = (pred_np > threshold).astype(int)
    
    # Compute metrics
    tp = np.sum((pred_binary == 1) & (target_np == 1))
    fp = np.sum((pred_binary == 1) & (target_np == 0))
    fn = np.sum((pred_binary == 0) & (target_np == 1))
    tn = np.sum((pred_binary == 0) & (target_np == 0))
    
    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    
    # Mean Average Precision (simplified)
    ap = precision  # Simplified version
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'ap': ap,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def create_interaction_matrix(
    num_objects: int,
    num_interactions: int,
    interactions: List[Tuple[int, int, int]]
) -> np.ndarray:
    """Create interaction matrix from interaction list.
    
    Args:
        num_objects: Number of objects
        num_interactions: Number of interaction types
        interactions: List of (subject_idx, object_idx, interaction_idx) tuples
        
    Returns:
        Interaction matrix [num_objects, num_objects, num_interactions]
    """
    matrix = np.zeros((num_objects, num_objects, num_interactions))
    
    for subject_idx, object_idx, interaction_idx in interactions:
        if (subject_idx < num_objects and 
            object_idx < num_objects and 
            interaction_idx < num_interactions):
            matrix[subject_idx, object_idx, interaction_idx] = 1.0
    
    return matrix


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array [H, W, C]
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image: np.ndarray, save_path: Union[str, Path]) -> None:
    """Save image to file.
    
    Args:
        image: Image as numpy array [H, W, C]
        save_path: Path to save image
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    cv2.imwrite(str(save_path), image_bgr)
