import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from dataset import Dataset
from model import VQVAE,Predictor
from utils import EarlyStopping

VAE_weight_path = r'/openbayes/home/Attention'
MLP_weight_path = r'/openbayes/home/Attention' 
save_path = r'/openbayes/home/Attention/metrics/mlp'
train_folder = r"/openbayes/home/dataset/SDV7/train"
val_45_folder = r"/openbayes/home/dataset/SDV7/val_45"
val_in_folder = r"/openbayes/home/dataset/SDV7/val_in"

num_workers = 18
prefetch_factor = 2
persistent_workers = True

batch_size = 32
embedding_dim = 16
num_embeddings = 512
latent_weight = 1.0
num_epochs = 1400

loader_kwargs = {
    'batch_size': batch_size,
    'num_workers': num_workers,
    'prefetch_factor': prefetch_factor,
    'persistent_workers': persistent_workers,
    'pin_memory': True
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

early_stopping = EarlyStopping(
        patience=1500000,
        save_path=os.path.join(save_path, "best_MLP_model.pth")
    )

VAE_model = VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
MLP_model = Predictor(embedding_dim=embedding_dim).to(device)

train_dataset = Dataset(train_folder, is_train=True)
val_45_dataset = Dataset(
    val_45_folder,
    min_vals=train_dataset.min_vals,
    max_vals=train_dataset.max_vals,
    is_train=False
)
val_in_dataset = Dataset(
    val_in_folder,
    min_vals=train_dataset.min_vals,
    max_vals=train_dataset.max_vals,
    is_train=False
)

train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
val_45_loader = DataLoader(val_45_dataset, shuffle=False, **loader_kwargs)
val_in_loader = DataLoader(val_in_dataset, shuffle=False, **loader_kwargs)

optimizer = torch.optim.Adam(
   MLP_model.parameters(), 
    lr=1e-3,
    betas=(0.9, 0.999)
)

scheduler = StepLR(optimizer, 
                step_size=500,
                gamma=0.1)

mse_loss = nn.MSELoss()

def VAE_load_weights(model, VAE_weight_path):
    best_model_path = os.path.join(VAE_weight_path, "best_vqvae_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"VAE Loaded best model weights from: {best_model_path}")
    else:
        print("VAE no previous weights found. Starting training from scratch.")
    return model

def MLP_load_weights(model, MLP_weight_path):
    best_model_path = os.path.join(MLP_weight_path, "best_MLP_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"MLP Loaded best model weights from: {best_model_path}")
    else:
        print("MLP no previous weights found. Starting training from scratch.")
    return model

VAE_model = VAE_load_weights(VAE_model, VAE_weight_path)
MLP_model = MLP_load_weights(MLP_model,MLP_weight_path)

def train_epoch(VAE_model,MLP_model, dataloader, optimizer, device):
    VAE_model.eval()
    MLP_model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_latent_loss = 0.0

    for image,vector in tqdm(dataloader, desc="Training"):
        vector = vector.to(device)
        image = image.to(device)

        optimizer.zero_grad()
        latent = MLP_model(vector)
        _,quantized,_,_= VAE_model.encode(image,vector)
            
        pre_recon = VAE_model.decode(latent)
        vae_recon = VAE_model.decode(quantized)

        loss_latent = torch.nn.functional.mse_loss(quantized,latent)
        loss_recons = torch.nn.functional.mse_loss(pre_recon, vae_recon)
        loss = loss_recons + loss_latent * latent_weight
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon_loss += loss_recons.item()
        total_latent_loss += loss_latent.item()

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_latent_loss = total_latent_loss / len(dataloader)
    return avg_loss, avg_recon_loss, avg_latent_loss

def validate_epoch(VAE_model,MLP_model,dataloader, device):
    VAE_model.eval()
    MLP_model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_latent_loss = 0.0

    with torch.no_grad():
        for image, vector in tqdm(dataloader, desc="Validation"):
            vector = vector.to(device)
            image = image.to(device)

            optimizer.zero_grad()

            latent = MLP_model(vector)
            _,quantized,_,_ = VAE_model.encode(image,vector)
            
            pre_recon = VAE_model.decode(latent)
            vae_recon = VAE_model.decode(quantized)

            loss_latent = torch.nn.functional.mse_loss(quantized,latent)
            loss_recons = torch.nn.functional.mse_loss(pre_recon, vae_recon)
            loss = loss_recons + loss_latent * latent_weight

            total_loss += loss.item()
            total_recon_loss += loss_recons.item()
            total_latent_loss += loss_latent.item()

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_latent_loss = total_latent_loss / len(dataloader)

    return avg_loss, avg_recon_loss, avg_latent_loss

if __name__ == '__main__':
    train_losses = {'avg_loss': [], 'avg_recon_loss': [], 'avg_latent_loss': []}
    val_45_losses = {'avg_loss': [], 'avg_recon_loss': [], 'avg_latent_loss': []}
    val_in_losses = {'avg_loss': [], 'avg_recon_loss': [], 'avg_latent_loss': []}
    
    if not os.path.exists(os.path.join(save_path, "MLP_loss_log.csv")):
        os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "MLP_loss_log.csv"), 'w') as f:
        pass
    
    log_file_path = os.path.join(save_path, "MLP_loss_log.csv")
    
    headers = [
    "epoch",
    "train_loss", "train_recon", "train_latent",
    "val_45_loss", "val_45_recon", "val_45_latent",
    "val_in_loss", "val_in_recon", "val_in_latent"
]
    
    
    with open(log_file_path, "w") as f:
        f.write(",".join(headers) + "\n")

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss, train_recon, train_latent = train_epoch(VAE_model,MLP_model, train_loader, optimizer, device)
        scheduler.step()
        
        val_45, val_45_recon, val_45_latent = validate_epoch(VAE_model,MLP_model, val_45_loader, device)
        
        val_in, val_in_recon, val_in_latent = validate_epoch(VAE_model,MLP_model, val_in_loader, device)
        
        train_losses['avg_loss'].append(train_loss)
        train_losses['avg_recon_loss'].append(train_recon)
        train_losses['avg_latent_loss'].append(train_latent)

        val_45_losses['avg_loss'].append(val_45)
        val_45_losses['avg_recon_loss'].append(val_45_recon)
        val_45_losses['avg_latent_loss'].append(val_45_latent)
        
        val_in_losses['avg_loss'].append(val_in)
        val_in_losses['avg_recon_loss'].append(val_in_recon)
        val_in_losses['avg_latent_loss'].append(val_in_latent)
        
        
        with open(log_file_path, "a") as f:
            metrics = f"""{epoch},\
        {train_loss:.4f},{train_recon:.4f},{train_latent:.4f},\
        {val_45:.4f},{val_45_recon:.4f},{val_45_latent:.4f},\
        {val_in:.4f},{val_in_recon:.4f},{val_in_latent:.4f}\n"""
            f.write(metrics)
        
        if early_stopping(val_45, val_in, MLP_model):
            break