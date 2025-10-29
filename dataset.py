import os
import json
import re
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class _Dataset(Dataset):
    def __init__(self, json_file, image_dir, split='train', transform=None, min_vals=None, max_vals=None):
        """
        Initialize the dataset

        Parameters:

        json_file (str): Path to the JSON file containing the split dataset

        image_dir (str): Path to the directory containing the image files

        split (str): 'train' or 'val', specifying whether to use the training or validation set

        transform (callable, optional): Optional transformations applied to the images

        min_vals (tuple, optional): The minimum value of the feature vectors, used for normalization. If None, it is calculated from the training data.

        max_vals (tuple, optional): The maximum value of the feature vectors, used for normalization. If None, it is calculated from the training data.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.split = split
        self.data = []
        
        with open(json_file, 'r') as f:
            dataset = json.load(f)
        
        if split not in dataset:
            raise ValueError(f"JSON file does not contain '{split}' split")
        
        for entry in dataset[split]:
            parsed = self._parse_entry(entry)
            if parsed:
                self.data.append(parsed)
        
        print(f"Loaded {len(self.data)} entries from {split} sets")
        
        if min_vals is None or max_vals is None:
            self._compute_normalization_params()
        else:
            self.min_vals = min_vals
            self.max_vals = max_vals
    
    def _parse_entry(self, entry):
        """Parse a data entry, extracting name, y, x, n, d, and four-dimensional one-hot encoding"""
        pattern = r"^(\d+),(\d+),(\d+),(\d+),(-?\d+),'([^']+)'$"
        match = re.match(pattern, entry)
        
        if not match:
            print(f"Warning: Could not parse entry: {entry}")
            return None
        
        name, y, x, n, d, one_hot_str = match.groups()

        # Convert to appropriate types
        name = int(name)
        y = int(y)
        x = int(x)
        n = int(n)
        d = int(d)

        # Parse four-dimensional one-hot encoding
        one_hot = [float(val) for val in one_hot_str.split(',')]

        # Validate one-hot encoding dimension
        if len(one_hot) != 4:
            print(f"Warning: One-hot encoding '{one_hot_str}' for entry {name} is not four-dimensional")
        
        return {
            'name': name,
            'features': (y, x, n, d),
            'one_hot': one_hot,
            'entry': entry
        }
    
    def _compute_normalization_params(self):
        """Compute the min and max values of the feature vectors for MinMax normalization"""
        # Extract all feature vectors
        all_features = np.array([item['features'] for item in self.data])

        # Compute the min and max values for each dimension
        self.min_vals = tuple(np.min(all_features, axis=0))
        self.max_vals = tuple(np.max(all_features, axis=0))

        print(f"Computed normalization parameters - Min: {self.min_vals}, Max: {self.max_vals}")

    def normalize_features(self, features):
        """Normalize the feature vector using MinMax scaling"""
        normalized = []
        for i in range(len(features)):
            # Avoid division by zero
            if self.max_vals[i] == self.min_vals[i]:
                normalized.append(0.0)  # If max == min, normalize to 0
            else:
                normalized.append(
                    (features[i] - self.min_vals[i]) / (self.max_vals[i] - self.min_vals[i])
                )
        return tuple(normalized)
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Returns:
            image: Image tensor
            features: Normalized four-dimensional feature vector (y,x,n,d)
            one_hot: Four-dimensional one-hot encoding
        """
        item = self.data[idx]
        name = item['name']

        img_path = os.path.join(self.image_dir, f"{name}.png")

        if not os.path.exists(img_path):
            for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
                alt_path = os.path.join(self.image_dir, f"{name}{ext}")
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            
            raw_features = item['features']
            normalized_features = self.normalize_features(raw_features)
            features = torch.tensor(normalized_features, dtype=torch.float)
            one_hot = torch.tensor(item['one_hot'], dtype=torch.float)
            
            return image, features, one_hot
            
        except Exception as e:
            print(f"加载图像 {img_path} 时出错: {e}")
            placeholder_image = torch.zeros((3, 224, 224))
            features = torch.tensor(self.normalize_features(item['features']), dtype=torch.float)
            one_hot = torch.tensor(item['one_hot'], dtype=torch.float)
            
            return placeholder_image, features, one_hot
