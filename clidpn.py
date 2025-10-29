import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataset import _Dataset
from model_ab import VQVAE
from utils import VQLossTracker,CheckpointSaver,create_filtered_loader
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight_path = r'/openbayes/home/CrossMaterial'
save_path = r'/openbayes/home/CrossMaterial/metrics/vqvae'
loss_save_path = r'/openbayes/home/CrossMaterial/metrics/vqvae'
os.makedirs(weight_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(loss_save_path, exist_ok=True)

json_file = r"/openbayes/input/input0/data.json"
image_dir = r"/openbayes/input/input0"

batch_size = 32
num_workers = 18
embedding_dim = 16
num_embeddings = 512
num_epochs = 500

train_dataset = _Dataset(json_file=json_file, image_dir=image_dir, split='train')
val_dataset = _Dataset(json_file=json_file, image_dir=image_dir, split='val')

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

val_1000_loader, val_1000_count = create_filtered_loader(val_dataset, [1.0, 0.0, 0.0, 0.0], "val_1000")
val_0100_loader, val_0100_count = create_filtered_loader(val_dataset, [0.0, 1.0, 0.0, 0.0], "val_0100")
val_0010_loader, val_0010_count = create_filtered_loader(val_dataset, [0.0, 0.0, 1.0, 0.0], "val_0010")
val_0001_loader, val_0001_count = create_filtered_loader(val_dataset, [0.0, 0.0, 0.0, 1.0], "val_0001")

criterion = nn.MSELoss()

model = VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-3,
    betas=(0.9, 0.999)
)

scheduler = StepLR(
    optimizer,
    step_size=200,
    gamma=0.1
)


def load_weights(model, weight_path):
    best_model_path = os.path.join(weight_path, "best_vqvae_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model weights from: {best_model_path}")
    else:
        print("No previous weights found. Starting training from scratch.")
    return model

model = load_weights(model, weight_path)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    
    for image, feat, one_hot in tqdm(dataloader, desc="Training"):
        image = image.to(device)
        feat = feat.to(device)
        one_hot = one_hot.to(device)

        optimizer.zero_grad()
        
        loss_vq, quantized, perplexity, encodings = model.encode(image, feat, one_hot)
        x_recon = model.decode(quantized)

        loss_recons = criterion(x_recon, image) / model.data_variance
        
        loss = loss_recons + loss_vq

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += loss_recons.item()
        total_vq_loss += loss_vq.item()
        
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_vq_loss = total_vq_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_vq_loss

def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    
    if len(dataloader) == 0:
        return 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for image, feat, one_hot in tqdm(dataloader, desc="Validation"):
            image = image.to(device)
            feat = feat.to(device)
            one_hot = one_hot.to(device) 

            loss_vq, quantized, perplexity, encodings = model.encode(image, feat, one_hot)
            x_recon = model.decode(quantized)

            loss_recons = criterion(x_recon, image) / model.data_variance
            
            loss = loss_recons + loss_vq
            
            total_loss += loss.item()
            total_recon_loss += loss_recons.item()
            total_vq_loss += loss_vq.item()
            
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_vq_loss = total_vq_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_vq_loss

if __name__ == '__main__':
    checkpoint_saver = CheckpointSaver(save_path, save_interval=200)
    loss_tracker = VQLossTracker(loss_save_path)
                
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        
        val_1000_metrics = validate_epoch(model, val_1000_loader, device)
        val_0100_metrics = validate_epoch(model, val_0100_loader, device)
        val_0010_metrics = validate_epoch(model, val_0010_loader, device)
        val_0001_metrics = validate_epoch(model, val_0001_loader, device)
        
        metrics_dict = {
            'train': train_metrics,
            'val_1000': val_1000_metrics,
            'val_0100': val_0100_metrics,
            'val_0010': val_0010_metrics,
            'val_0001': val_0001_metrics
        }
        
        loss_tracker.update(epoch, **metrics_dict)

        avg_val_loss = (val_1000_metrics[0] + val_0100_metrics[0] +
                        val_0010_metrics[0] + val_0001_metrics[0]  ) / 4
        
        checkpoint_saver.update(epoch, model, avg_val_loss)