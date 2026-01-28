#!/usr/bin/env python3
"""Training script for HOI detection."""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hoi_detection import HOIDetector, HICODataset, HOITrainer
from hoi_detection.utils import setup_device, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HOI detection model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/hico_det.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True,
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to use (for debugging)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config['device'] = args.device
    config['seed'] = args.seed
    config['data']['root'] = args.data_root
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup device
    device = setup_device(config['device'])
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = HICODataset(
        data_root=config['data']['root'],
        split=config['data']['train_split'],
        max_samples=args.max_samples
    )
    
    val_dataset = HICODataset(
        data_root=config['data']['root'],
        split=config['data']['val_split'],
        max_samples=args.max_samples
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model = HOIDetector(
        device=device,
        score_threshold=config['evaluation']['score_threshold'],
        backend="detectron2"
    )
    
    # Create trainer
    print("Setting up trainer...")
    trainer = HOITrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=config['training']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
