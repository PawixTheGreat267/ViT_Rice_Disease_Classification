from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch


def prepare_dataset(batch_size, data_dir, input_size):
    transform = transform_image(input_size)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_dataset, val_dataset = train_val_split(full_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def transform_image(input_size):
    mean, std = get_imagenet_mean_std()
    transform = transforms.Compose([
        transforms.Resize(input_size),  
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  
        ])
    return transform

def get_imagenet_mean_std():
    mean = [0.485, 0.456, 0.406]   
    std = [0.229, 0.224, 0.225] 
    return (mean, std) 


def train_val_split(data):
    # Extract labels from the dataset
    targets = np.array(data.targets)

    # Stratified split (80% train, 20% validation)
    split = StratifiedShuffleSplit(train_size=0.8)
    train_idx, val_idx = next(split.split(np.zeros(len(targets)), targets))
    
    # Create train and validation datasets using the indices
    train_dataset = torch.utils.data.Subset(data, train_idx)
    val_dataset = torch.utils.data.Subset(data, val_idx)
    
    return (train_dataset, val_dataset)

def get_test_loader(batch_size, test_dir, input_size):
    transform = transform_image(input_size)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader