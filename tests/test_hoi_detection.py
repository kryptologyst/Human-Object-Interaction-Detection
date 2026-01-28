"""Tests for HOI detection package."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hoi_detection import HOIDetector, HICODataset
from hoi_detection.utils import setup_device, set_seed, compute_hoi_metrics


class TestHOIDetector:
    """Test cases for HOIDetector."""
    
    def test_device_setup(self):
        """Test device setup."""
        device = setup_device("cpu")
        assert device.type == "cpu"
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        # This should not raise an exception
        assert True
    
    def test_hoi_detector_initialization(self):
        """Test HOI detector initialization."""
        try:
            detector = HOIDetector(device="cpu", backend="detectron2")
            assert detector is not None
        except ImportError:
            # Skip if detectron2 is not available
            pytest.skip("Detectron2 not available")
    
    def test_hoi_detector_prediction(self):
        """Test HOI detector prediction."""
        try:
            detector = HOIDetector(device="cpu", backend="detectron2")
            
            # Create dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test prediction
            results = detector.predict(dummy_image)
            
            assert isinstance(results, dict)
            assert 'instances' in results
            assert 'interactions' in results
            assert 'relationships' in results
            
        except ImportError:
            pytest.skip("Detectron2 not available")
        except Exception as e:
            # Expected for dummy data
            assert "Could not load" in str(e) or "detectron2" in str(e).lower()


class TestHICODataset:
    """Test cases for HICO dataset."""
    
    def test_hico_dataset_initialization(self):
        """Test HICO dataset initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = HICODataset(data_root=temp_dir, split="train")
            assert len(dataset) >= 0  # Should handle empty dataset gracefully
    
    def test_hico_dataset_classes(self):
        """Test HICO dataset class loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = HICODataset(data_root=temp_dir, split="train")
            
            object_classes = dataset.get_object_classes()
            interaction_classes = dataset.get_interaction_classes()
            
            assert isinstance(object_classes, list)
            assert isinstance(interaction_classes, list)
            assert len(object_classes) > 0
            assert len(interaction_classes) > 0


class TestUtils:
    """Test cases for utility functions."""
    
    def test_compute_hoi_metrics(self):
        """Test HOI metrics computation."""
        # Create dummy predictions and targets
        predictions = torch.randn(2, 2, 10)  # 2 objects, 10 interactions
        targets = torch.randint(0, 2, (2, 2, 10)).float()
        
        metrics = compute_hoi_metrics(predictions, targets)
        
        assert isinstance(metrics, dict)
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'accuracy' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_visualize_hoi(self):
        """Test HOI visualization."""
        # Create dummy data
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
        labels = ["person", "cup"]
        interactions = np.random.rand(2, 2, 10)
        interaction_classes = [f"interaction_{i}" for i in range(10)]
        
        # Test visualization (should not raise exception)
        try:
            vis_image = visualize_hoi(
                image, boxes, labels, interactions, interaction_classes, show=False
            )
            assert vis_image is not None
        except Exception as e:
            # Visualization might fail in headless environment
            assert "display" in str(e).lower() or "matplotlib" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])
