import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from dataset import _Dataset
from model_ab import VQVAE, Predictor
from utils import create_filtered_loader
from skimage.metrics import structural_similarity as ssim
import lpips
import math
from json import JSONEncoder

# Create a custom JSON encoder to handle NumPy and special types
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Modify the Dataset class to return IDs
class EnhancedDataset(_Dataset):
    def __getitem__(self, idx):
        image, features, one_hot = super().__getitem__(idx)
        name = self.data[idx]['name']
        return image, features, one_hot, name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAE_weight_path = r'/openbayes/home/CrossMaterial/metrics/vqvae/ab/SDV8'
MLP_weight_path = r'/openbayes/home/CrossMaterial/metrics/mlp/ab/SDV8'
json_file = r"/openbayes/input/input1/SDV8_data.json"
image_dir = r"/openbayes/input/input1"
evaluation_output_dir = r"/openbayes/home/CrossMaterial/evaluation_results/SDV8ab"

batch_size = 16
num_workers = 18
embedding_dim = 16
num_embeddings = 512

os.makedirs(evaluation_output_dir, exist_ok=True)

lpips_model = lpips.LPIPS(net='alex').to(device)

val_dataset = EnhancedDataset(json_file=json_file, image_dir=image_dir, split='val')

def create_enhanced_filtered_loader(dataset, target_onehot, name, batch_size=16, shuffle=False, num_workers=8):
    indices = []
    target_onehot = torch.tensor(target_onehot)
    
    for i in range(len(dataset)):
        _, _, one_hot, _ = dataset[i]
        if torch.all(one_hot == target_onehot):
            indices.append(i)
    
    loader = DataLoader(
        dataset, batch_size=batch_size, 
        sampler=torch.utils.data.SubsetRandomSampler(indices),
        num_workers=num_workers, pin_memory=True
    )
    
    return loader, len(indices)

val_1000_loader, val_1000_count = create_enhanced_filtered_loader(val_dataset, [1.0, 0.0, 0.0, 0.0], "val_1000")
val_0100_loader, val_0100_count = create_enhanced_filtered_loader(val_dataset, [0.0, 1.0, 0.0, 0.0], "val_0100")
val_0010_loader, val_0010_count = create_enhanced_filtered_loader(val_dataset, [0.0, 0.0, 1.0, 0.0], "val_0010")
val_0001_loader, val_0001_count = create_enhanced_filtered_loader(val_dataset, [0.0, 0.0, 0.0, 1.0], "val_0001")


VAE_model = VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
MLP_model = Predictor(embedding_dim=embedding_dim).to(device)

def load_model(model, weight_path, model_name):
    checkpoint_path = os.path.join(weight_path, f"best_{model_name}_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded {model_name} model weights from: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No weights found at {checkpoint_path}")
    return model

VAE_model = load_model(VAE_model, VAE_weight_path, "vqvae")
MLP_model = load_model(MLP_model, MLP_weight_path, "mlp")

VAE_model.eval()
MLP_model.eval()
lpips_model.eval()

def setup_output_directories(base_dir, dataset_names):
    dirs = {}
    for name in dataset_names:
        dirs[name] = {
            'original': os.path.join(base_dir, name, 'original'),
            'mlp_recon': os.path.join(base_dir, name, 'mlp_recon'),
            'comparison': os.path.join(base_dir, name, 'comparison')
        }
        
        for d in dirs[name].values():
            os.makedirs(d, exist_ok=True)
    
    return dirs

output_dirs = setup_output_directories(
    evaluation_output_dir, 
    ['val_1000', 'val_0100', 'val_0010', 'val_0001']
)

def compute_image_metrics(img1, img2, lpips_model=None):
    """
    Calculate RMSE, LPIPS, SSIM, and PSNR between two images.
    
    Parameters:
        img1, img2: Tensors of the two images [C,H,W]
        lpips_model: Pre-loaded LPIPS model
    
    Returns:
        A dictionary containing the four metrics
    """
    if torch.is_tensor(img1):
        img1_np = img1.detach().cpu().permute(1, 2, 0).numpy()
        img2_np = img2.detach().cpu().permute(1, 2, 0).numpy()
    else:
        img1_np = img1
        img2_np = img2
    
    # RMSE
    mse = np.mean((img1_np - img2_np) ** 2)
    rmse = np.sqrt(mse)
    
    # PSNR
    max_pixel = 1.0
    psnr_value = 10 * np.log10((max_pixel ** 2) / mse) if mse > 0 else 100.0
    
    # SSIM
    ssim_value = ssim(img1_np, img2_np, 
                      data_range=1.0, 
                      multichannel=True, 
                      channel_axis=2)
    
    # LPIPS
    lpips_value = None
    if lpips_model is not None:
        img1_tensor = img1.unsqueeze(0).to(device) if torch.is_tensor(img1) else torch.from_numpy(
            img1_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        img2_tensor = img2.unsqueeze(0).to(device) if torch.is_tensor(img2) else torch.from_numpy(
            img2_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        
        img1_tensor = 2 * img1_tensor - 1
        img2_tensor = 2 * img2_tensor - 1
        
        with torch.no_grad():
            lpips_value = lpips_model(img1_tensor, img2_tensor).item()
    
    return {
        'rmse': float(rmse),
        'lpips': float(lpips_value) if lpips_value is not None else None,
        'ssim': float(ssim_value),
        'psnr': float(psnr_value)
    }

def evaluate_dataset(vae_model, mlp_model, dataloader, output_dirs, dataset_name, lpips_model):
    """Evaluate the dataset and save the result image"""

    # Initialize cumulative metrics data
    metrics_data = {
        'dataset': dataset_name,
        'samples': [],
        'overall': {
            'original_vs_mlp': {'rmse': 0.0, 'lpips': 0.0, 'ssim': 0.0, 'psnr': 0.0}
        }
    }
    
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (images, features, one_hot, img_ids) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}")):
            images = images.to(device)
            features = features.to(device)
            one_hot = one_hot.to(device)
            
            mlp_latent = mlp_model(features, one_hot)
            
            mlp_recon = vae_model.decode(mlp_latent)
            
            for i in range(images.size(0)):
                img_id = int(img_ids[i].item())
                img_name = f"{img_id}.png"
                
                save_image(images[i], os.path.join(output_dirs[dataset_name]['original'], img_name))
                
                save_image(mlp_recon[i], os.path.join(output_dirs[dataset_name]['mlp_recon'], img_name))
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                orig_img = images[i].cpu().permute(1, 2, 0).numpy()
                mlp_img = mlp_recon[i].cpu().permute(1, 2, 0).numpy()
                
                axes[0].imshow(np.clip(orig_img, 0, 1))
                axes[0].set_title(f"Original (ID: {img_id})")
                axes[0].axis('off')
                
                axes[1].imshow(np.clip(mlp_img, 0, 1))
                axes[1].set_title("MLP Prediction")
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dirs[dataset_name]['comparison'], img_name))
                plt.close(fig)
                
                metrics_orig_mlp = compute_image_metrics(images[i], mlp_recon[i], lpips_model)
                
                sample_metrics = {
                    'image_id': img_id,
                    'image_name': img_name,
                    'original_vs_mlp': metrics_orig_mlp
                }
                metrics_data['samples'].append(sample_metrics)
                
                for metric in ['rmse', 'lpips', 'ssim', 'psnr']:
                    metrics_data['overall']['original_vs_mlp'][metric] += float(metrics_orig_mlp[metric])
                
                total_samples += 1
    
    if total_samples > 0:
        for metric in ['rmse', 'lpips', 'ssim', 'psnr']:
            metrics_data['overall']['original_vs_mlp'][metric] /= total_samples
    
    return metrics_data

def main():
    dataset_loaders = {
        'val_1000': val_1000_loader,
        'val_0100': val_0100_loader,
        'val_0010': val_0010_loader,
        'val_0001': val_0001_loader
    }
    
    all_metrics = {}
    
    for name, loader in dataset_loaders.items():
        metrics_data = evaluate_dataset(
            VAE_model, MLP_model, loader, output_dirs, name, lpips_model
        )

        all_metrics[name] = metrics_data

        print(f"\n{name} Evaluation Overall Metrics (Original vs MLP Prediction):")
        metrics = metrics_data['overall']['original_vs_mlp']
        print(f"  RMSE:  {metrics['rmse']:.6f}")
        print(f"  LPIPS: {metrics['lpips']:.6f}")
        print(f"  SSIM:  {metrics['ssim']:.6f}")
        print(f"  PSNR:  {metrics['psnr']:.6f}dB")

    # Save detailed metrics to JSON file
    json_path = os.path.join(evaluation_output_dir, 'detailed_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, cls=NumpyEncoder)  # 使用自定义编码器

    # Generate concise summary JSON
    summary_metrics = {}
    for dataset_name, data in all_metrics.items():
        summary_metrics[dataset_name] = data['overall']['original_vs_mlp']
    
    summary_path = os.path.join(evaluation_output_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_metrics, f, indent=2, cls=NumpyEncoder)

if __name__ == "__main__":
    main()
