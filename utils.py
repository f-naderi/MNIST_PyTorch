# utils.py
import os
import gzip
import struct
import requests
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

  
DATA_DIR = './mnist_data'
BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master"

def load_mnist_from_github(cache_dir=DATA_DIR):
    """
    Load dataset from GitHub. If files already exist in cache_dir,
    they are read from disk and no download is needed.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    raw_data = {}
    for name, filename in files.items():
        filepath = os.path.join(cache_dir, filename)
        
        # Download only if the file does not exist  
        if not os.path.exists(filepath):
            print(f"Downloading {filename} from GitHub...")
            url = f"{BASE_URL}/{filename}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Saved to {filepath}")
        else:
            print(f"Found {filename} on disk. Loading...")
        
        # Read file from disk    
        with gzip.open(filepath, 'rb') as f:
            raw_data[name] = f.read()

    
    def parse_images(raw):
        magic, num, rows, cols = struct.unpack('>IIII', raw[:16])
        return np.frombuffer(raw[16:], dtype=np.uint8).reshape(num, rows, cols)

    def parse_labels(raw):
        magic, num = struct.unpack('>II', raw[:8])
        return np.frombuffer(raw[8:], dtype=np.uint8)

    X_train = parse_images(raw_data['train_images'])
    y_train = parse_labels(raw_data['train_labels'])
    X_test = parse_images(raw_data['test_images'])
    y_test = parse_labels(raw_data['test_labels'])
    
    return (X_train, y_train), (X_test, y_test)


class MNISTDataset(Dataset):
    # Custom PyTorch Dataset for MNIST
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Add channel dimension (1, 28, 28)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default normalization 
            image = image / 255.0
        
        return image, label


def get_data_loaders(batch_size=64):
    # Create training and test DataLoaders
    print("\nLoading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_mnist_from_github()
    
    print(f"Training data: {X_train.shape}, Training labels: {y_train.shape}")
    print(f"Test data: {X_test.shape}, Test labels: {y_test.shape}")
    
    train_dataset = MNISTDataset(X_train, y_train)
    test_dataset = MNISTDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_device():
    # Detect and return the appropriate device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    return device