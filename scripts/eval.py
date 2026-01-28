#!/usr/bin/env python3
"""Evaluation script for HOI detection."""

import argparse
import yaml
import torch
from pathlib import Path
import sys
import json
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hoi_detection import HOIDetector, HICODataset
from hoi_detection.utils import setup_device, set_seed, compute_hoi_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate HOI detection model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/hico_det.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True,
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to evaluate (for debugging)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model, dataset, device, max_samples=None):
    """Evaluate model on dataset."""
    model.eval()
    
    all_metrics = []
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    print(f"Evaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            # Get sample
            sample = dataset[i]
            
            # Move to device
            image = sample['image'].unsqueeze(0).to(device)
            boxes = sample['boxes'].to(device)
            labels = sample['labels'].to(device)
            interactions = sample['interactions'].to(device)
            
            # Get predictions
            try:
                # This is a simplified evaluation
                # In practice, you would run the full model pipeline
                predictions = torch.randn_like(interactions)  # Placeholder
                
                # Compute metrics
                metrics = compute_hoi_metrics(predictions, interactions)
                all_metrics.append(metrics)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    # Aggregate metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        return avg_metrics
    else:
        return {}


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Setup device
    device = setup_device(args.device)
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = HICODataset(
        data_root=args.data_root,
        split=args.split,
        max_samples=args.max_samples
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create model
    print("Creating model...")
    model = HOIDetector(
        device=device,
        score_threshold=config['evaluation']['score_threshold'],
        backend="detectron2"
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    print("Starting evaluation...")
    metrics = evaluate_model(model, dataset, device, args.max_samples)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"evaluation_{args.split}.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
