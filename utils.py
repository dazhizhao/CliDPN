import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader,Subset
import collections
import pandas as pd

class EarlyStopping:
    def __init__(self, patience=10, delta=0, save_path="best_model.pth"):
        """
        Args:
            patience (int): Number of validation loss rounds tolerated without boosting
            delta (float): Minimum change considered a boost
            save_path (str): Model save path
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path

        # Track best losses for two validation sets and total loss
        self.best_loss1 = None
        self.best_loss2 = None
        self.best_sum_loss = None
        self.counter = 0
        
    def __call__(self, val_loss1, val_loss2, model):
        """
        Args:
            val_loss1 (float): First validation set loss
            val_loss2 (float): Second validation set loss
            model (nn.Module): Current model

        Returns:
            bool: Whether to trigger early stopping (True means stop training)
        """
        current_sum_loss = val_loss1 + val_loss2
        should_save = False

        if self.best_loss1 is None:
            self.best_loss1 = val_loss1
            self.best_loss2 = val_loss2
            self.best_sum_loss = current_sum_loss
            should_save = True
        else:
            if val_loss1 < self.best_loss1 - self.delta:
                self.best_loss1 = val_loss1
                should_save = True
                
            if val_loss2 < self.best_loss2 - self.delta:
                self.best_loss2 = val_loss2
                should_save = True
        
        if should_save:
            self.save_best_model(model)
            print(f"Saved model!")
        
        if current_sum_loss < self.best_sum_loss - self.delta:
            self.best_sum_loss = current_sum_loss
            self.counter = 0
        else:
            self.counter += 1
    
        if self.counter >= self.patience:
            print("Trigged EarlyStopping!")
            return True
            
        return False

    def save_best_model(self, model):
        torch.save(model.state_dict(), self.save_path)

class VQLossTracker:
    def __init__(self, save_dir):
        """Initialize loss tracker
        
        Args:
            save_dir: Directory path to save loss records
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics = {
            'epochs': [],
            'metrics': collections.defaultdict(list)
        }

        # Define metric types and datasets to track
        self.metric_types = ['loss', 'recon_loss', 'vq_loss']
        self.datasets = ['train', 'val_1000', 'val_0100', 'val_0010', 'val_0001']
    
    def update(self, epoch, **metrics_dict):
        """Update various metrics
        
        Args:
            epoch: Current training epoch
            **metrics_dict: Dictionary containing various metrics, format:
                         {'train': (loss, recon_loss, vq_loss),
                          'val_1000': (loss, recon_loss, vq_loss),
                          ...}
        """
        self.metrics['epochs'].append(epoch)
        
        for dataset, values in metrics_dict.items():
            if not isinstance(values, (list, tuple)) or len(values) != 3:
                print(f"Warning: {dataset} metric format is incorrect, should be (loss, recon_loss, vq_loss)")
                continue
                
            loss, recon_loss, vq_loss = values

            # Save each type of metric
            for i, metric_type in enumerate(self.metric_types):
                metric_name = f"{dataset}_{metric_type}"
                self.metrics['metrics'][metric_name].append(values[i])

        self._save_losses()
    
    def _save_losses(self):
        """Save tracked losses as CSV file"""
        data = {'epoch': self.metrics['epochs']}
        
        for metric_name, values in self.metrics['metrics'].items():
            data[metric_name] = values
            
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.save_dir, 'loss_tracker.csv')
        df.to_csv(csv_path, index=False)
        
    def get_latest_metrics(self):
        """Get all metrics from the latest epoch"""
        if not self.metrics['epochs']:
            return None
            
        latest = {}
        for metric_name, values in self.metrics['metrics'].items():
            if values:
                latest[metric_name] = values[-1]
        
        return latest

class CheckpointSaver:
    def __init__(self, save_dir, save_interval=100):
        """
        Initialize checkpoint saver
        
        Args:
            save_dir (str): Directory path to save checkpoints
            save_interval (int): Save interval (how many epochs to save once)
        """
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.best_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)
        
    def update(self, epoch, model, loss):
        """
        Update checkpoint
        """
        is_best = loss < self.best_loss
        
        if epoch % self.save_interval == 0:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
        
        if is_best:
            self.best_loss = loss
            best_model_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with loss {loss:.6f} to {best_model_path}")
            
        return is_best

class MLPLossTracker:
    """
    Simplified MLP loss tracker, only retains CSV saving functionality, saving loss values with specified precision
    """
    def __init__(self, save_dir):
        """
        Initialize loss tracker

        Args:
            save_dir (str): Directory path to save loss records
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics = {
            'epochs': [],
            'metrics': collections.defaultdict(list)
        }
        
        self.metric_types = ['loss', 'recon_loss', 'latent_loss']
        self.datasets = ['train', 'val_1000', 'val_0100', 'val_0010', 'val_0001']

    def update(self, epoch, **metrics_dict):
        """
        Update all metrics
        
        Args:
            epoch (int): Current training epoch
            **metrics_dict: Dictionary containing various metrics, formatted as:
                {
                    'train': (loss, recon_loss, latent_loss),
                    'val_1000': (loss, recon_loss, latent_loss),
                    ...
                }
        """
        self.metrics['epochs'].append(epoch)
        
        for dataset, values in metrics_dict.items():
            if dataset not in self.datasets:
                continue
            
            if not isinstance(values, (list, tuple)) or len(values) != 3:
                continue
            
            for i, metric_type in enumerate(self.metric_types):
                metric_name = f"{dataset}_{metric_type}"
                self.metrics['metrics'][metric_name].append(values[i])
        
        if all(f"val_{suffix}_loss" in self.metrics['metrics'] for suffix in ['1000', '0100', '0010', '0001']):
            val_losses = [
                self.metrics['metrics'][f"val_{suffix}_loss"][-1] 
                for suffix in ['1000', '0100', '0010', '0001']
            ]
            avg_val_loss = sum(val_losses) / 4
            
            metric_name = "avg_val_loss"
            if len(self.metrics['metrics'].get(metric_name, [])) < len(self.metrics['epochs']):
                self.metrics['metrics'][metric_name].append(avg_val_loss)
            else:
                self.metrics['metrics'][metric_name][-1] = avg_val_loss
        
        self._save_losses()

    def _save_losses(self):
        """Save all losses in the dictionary as a CSV file, using the same precision as the example"""
        data = {'epoch': self.metrics['epochs']}
        
        for metric_name, values in self.metrics['metrics'].items():
            data[metric_name] = values
            
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.save_dir, 'mlp_loss_tracker.csv')
        df.to_csv(csv_path, index=False, float_format='%.16f')
        
    def get_latest_metrics(self):
        if not self.metrics['epochs']:
            return None
        
        latest = {}
        for metric_name, values in self.metrics['metrics'].items():
            if values:
                latest[metric_name] = values[-1]
        return latest


def create_filtered_loader(dataset, one_hot_filter, name=""):
    indices = []
    batch_size = 32
    num_workers = 18
    one_hot_counts = {}
    
    for i in range(len(dataset)):
        _, _, one_hot = dataset[i]
        one_hot_tuple = tuple(one_hot.tolist())
        one_hot_filter_tuple = tuple(one_hot_filter)
        
        if one_hot_tuple not in one_hot_counts:
            one_hot_counts[one_hot_tuple] = 0
        one_hot_counts[one_hot_tuple] += 1
        
        if np.allclose(one_hot_tuple, one_hot_filter_tuple, atol=1e-5):
            indices.append(i)

    print(f"\n=== {name} One-Hot Statistics ===")
    for encoding, count in one_hot_counts.items():
        print(f"  {encoding}: {count} samples")

    if len(indices) == 0:
        print(f"Warning: No samples found matching [{', '.join(map(str, one_hot_filter))}] in {name}!")
    else:
        print(f"Found {len(indices)} samples matching [{', '.join(map(str, one_hot_filter))}] in {name}.")

    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ), len(indices)