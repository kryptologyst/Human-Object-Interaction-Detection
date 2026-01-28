"""Training utilities for HOI detection."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path

from ..models import HOIDetector
from ..data import HOIDataset
from ..utils import compute_hoi_metrics, set_seed


class HOITrainer:
    """Trainer class for HOI detection models.
    
    Args:
        model: HOI detection model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: Device to train on
        config: Training configuration
    """
    
    def __init__(
        self,
        model: HOIDetector,
        train_dataset: HOIDataset,
        val_dataset: Optional[HOIDataset] = None,
        device: torch.device = torch.device("cuda"),
        config: Optional[Dict] = None
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.config = config or self._default_config()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss functions
        self.criterion = self._setup_criterion()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup logging
        if self.config.get('use_wandb', False):
            wandb.init(project="hoi-detection", config=self.config)
    
    def _default_config(self) -> Dict:
        """Default training configuration."""
        return {
            'learning_rate': 1e-4,
            'batch_size': 8,
            'num_epochs': 100,
            'weight_decay': 1e-4,
            'scheduler_step_size': 30,
            'scheduler_gamma': 0.1,
            'save_interval': 10,
            'val_interval': 5,
            'use_wandb': False,
            'checkpoint_dir': 'checkpoints'
        }
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler.StepLR:
        """Setup learning rate scheduler."""
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['scheduler_step_size'],
            gamma=self.config['scheduler_gamma']
        )
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss function."""
        return nn.BCEWithLogitsLoss()
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=4
        )
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get predictions
            outputs = self.model.predict_batch(batch['images'])
            
            # Compute loss
            loss = self._compute_loss(outputs, batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({'train_loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        if self.val_dataset is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=4
        )
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model.predict_batch(batch['images'])
                
                # Compute loss
                loss = self._compute_loss(outputs, batch)
                total_loss += loss.item()
                
                # Compute metrics
                metrics = self._compute_metrics(outputs, batch)
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value
                
                num_batches += 1
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        self.val_losses.append(avg_loss)
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            log_dict = {'val_loss': avg_loss}
            log_dict.update(avg_metrics)
            wandb.log(log_dict)
        
        return avg_loss
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                device_batch[key] = [v.to(self.device) for v in value]
            else:
                device_batch[key] = value
        return device_batch
    
    def _compute_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Compute training loss."""
        # This is a simplified implementation
        # In practice, you would compute losses for different components
        
        if 'interactions' in outputs and 'interactions' in batch:
            pred_interactions = outputs['interactions']
            target_interactions = batch['interactions']
            
            # Flatten for loss computation
            pred_flat = pred_interactions.view(-1)
            target_flat = target_interactions.view(-1)
            
            loss = self.criterion(pred_flat, target_flat)
        else:
            # Dummy loss if no interactions
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return loss
    
    def _compute_metrics(self, outputs: Dict, batch: Dict) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        if 'interactions' in outputs and 'interactions' in batch:
            pred_interactions = outputs['interactions']
            target_interactions = batch['interactions']
            
            # Compute HOI metrics
            hoi_metrics = compute_hoi_metrics(pred_interactions, target_interactions)
            metrics.update(hoi_metrics)
        
        return metrics
    
    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = Path(path).parent / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
    
    def train(self) -> None:
        """Train the model."""
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        
        # Create checkpoint directory
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            if epoch % self.config['val_interval'] == 0:
                val_loss = self.validate()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        checkpoint_dir / f'epoch_{epoch}.pth',
                        is_best=True
                    )
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(checkpoint_dir / f'epoch_{epoch}.pth')
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        print("Training completed!")
        
        # Close wandb
        if self.config.get('use_wandb', False):
            wandb.finish()
