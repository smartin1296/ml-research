import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional, Callable, List
import numpy as np
from pathlib import Path
import os

class CIFAR10Dataset:
    """
    CIFAR-10 dataset wrapper with standard preprocessing
    """
    
    def __init__(self, data_dir: str = './data/raw/images', 
                 normalize: bool = True, augment_train: bool = True):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.augment_train = augment_train
        
        # CIFAR-10 statistics
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        
    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get train and test transforms"""
        
        # Base transforms
        base_transforms = [transforms.ToTensor()]
        
        if self.normalize:
            base_transforms.append(transforms.Normalize(self.mean, self.std))
        
        # Test transforms (no augmentation)
        test_transform = transforms.Compose(base_transforms)
        
        # Train transforms (with optional augmentation)
        train_transforms = []
        
        if self.augment_train:
            train_transforms.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ])
        
        train_transforms.extend(base_transforms)
        train_transform = transforms.Compose(train_transforms)
        
        return train_transform, test_transform
    
    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Get train and test datasets"""
        train_transform, test_transform = self.get_transforms()
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=str(self.data_dir), train=True, download=True, transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=str(self.data_dir), train=False, download=True, transform=test_transform
        )
        
        return train_dataset, test_dataset
    
    def get_dataloaders(self, batch_size: int = 128, num_workers: int = None,
                       pin_memory: bool = True, persistent_workers: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Get train and test dataloaders"""
        train_dataset, test_dataset = self.get_datasets()
        
        # Auto-detect optimal num_workers if not specified
        if num_workers is None:
            import os
            num_workers = min(8, os.cpu_count() or 1)  # Cap at 8 for stability
        
        # Persistent workers only if num_workers > 0
        use_persistent = persistent_workers and num_workers > 0
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
            persistent_workers=use_persistent, prefetch_factor=2 if num_workers > 0 else 2
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=use_persistent, prefetch_factor=2 if num_workers > 0 else 2
        )
        
        return train_loader, test_loader

class CIFAR100Dataset:
    """
    CIFAR-100 dataset wrapper with standard preprocessing
    """
    
    def __init__(self, data_dir: str = './data/raw/images', 
                 normalize: bool = True, augment_train: bool = True):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.augment_train = augment_train
        
        # CIFAR-100 statistics (similar to CIFAR-10)
        self.mean = (0.5071, 0.4865, 0.4409)
        self.std = (0.2673, 0.2564, 0.2762)
        
    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get train and test transforms"""
        
        # Base transforms
        base_transforms = [transforms.ToTensor()]
        
        if self.normalize:
            base_transforms.append(transforms.Normalize(self.mean, self.std))
        
        # Test transforms (no augmentation)
        test_transform = transforms.Compose(base_transforms)
        
        # Train transforms (with optional augmentation)
        train_transforms = []
        
        if self.augment_train:
            train_transforms.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
            ])
        
        train_transforms.extend(base_transforms)
        train_transform = transforms.Compose(train_transforms)
        
        return train_transform, test_transform
    
    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Get train and test datasets"""
        train_transform, test_transform = self.get_transforms()
        
        train_dataset = torchvision.datasets.CIFAR100(
            root=str(self.data_dir), train=True, download=True, transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR100(
            root=str(self.data_dir), train=False, download=True, transform=test_transform
        )
        
        return train_dataset, test_dataset
    
    def get_dataloaders(self, batch_size: int = 128, num_workers: int = None,
                       pin_memory: bool = True, persistent_workers: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Get train and test dataloaders"""
        train_dataset, test_dataset = self.get_datasets()
        
        # Auto-detect optimal num_workers if not specified
        if num_workers is None:
            import os
            num_workers = min(8, os.cpu_count() or 1)  # Cap at 8 for stability
        
        # Persistent workers only if num_workers > 0
        use_persistent = persistent_workers and num_workers > 0
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
            persistent_workers=use_persistent, prefetch_factor=2 if num_workers > 0 else 2
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=use_persistent, prefetch_factor=2 if num_workers > 0 else 2
        )
        
        return train_loader, test_loader

class MNISTDataset:
    """
    MNIST dataset wrapper for CNN testing on simpler data
    """
    
    def __init__(self, data_dir: str = './data/raw/images', 
                 normalize: bool = True, augment_train: bool = False):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.augment_train = augment_train
        
        # MNIST statistics
        self.mean = (0.1307,)
        self.std = (0.3081,)
    
    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get train and test transforms"""
        
        # Base transforms
        base_transforms = [transforms.ToTensor()]
        
        if self.normalize:
            base_transforms.append(transforms.Normalize(self.mean, self.std))
        
        # Test transforms (no augmentation)
        test_transform = transforms.Compose(base_transforms)
        
        # Train transforms (with optional augmentation)
        train_transforms = []
        
        if self.augment_train:
            train_transforms.extend([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
        
        train_transforms.extend(base_transforms)
        train_transform = transforms.Compose(train_transforms)
        
        return train_transform, test_transform
    
    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Get train and test datasets"""
        train_transform, test_transform = self.get_transforms()
        
        train_dataset = torchvision.datasets.MNIST(
            root=str(self.data_dir), train=True, download=True, transform=train_transform
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root=str(self.data_dir), train=False, download=True, transform=test_transform
        )
        
        return train_dataset, test_dataset
    
    def get_dataloaders(self, batch_size: int = 128, num_workers: int = None,
                       pin_memory: bool = True, persistent_workers: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Get train and test dataloaders"""
        train_dataset, test_dataset = self.get_datasets()
        
        # Auto-detect optimal num_workers if not specified
        if num_workers is None:
            import os
            num_workers = min(8, os.cpu_count() or 1)  # Cap at 8 for stability
        
        # Persistent workers only if num_workers > 0
        use_persistent = persistent_workers and num_workers > 0
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
            persistent_workers=use_persistent, prefetch_factor=2 if num_workers > 0 else 2
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=use_persistent, prefetch_factor=2 if num_workers > 0 else 2
        )
        
        return train_loader, test_loader

class ImageDatasetFactory:
    """
    Factory for creating different image datasets
    """
    
    @staticmethod
    def create_dataset(dataset_name: str, **kwargs):
        """
        Create a dataset by name
        
        Args:
            dataset_name: 'cifar10', 'cifar100', 'mnist'
            **kwargs: Additional arguments for dataset constructor
            
        Returns:
            Dataset instance
        """
        
        if dataset_name.lower() == 'cifar10':
            return CIFAR10Dataset(**kwargs)
        elif dataset_name.lower() == 'cifar100':
            return CIFAR100Dataset(**kwargs)
        elif dataset_name.lower() == 'mnist':
            return MNISTDataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    @staticmethod
    def get_dataset_info(dataset_name: str) -> dict:
        """Get information about a dataset"""
        
        info = {
            'cifar10': {
                'num_classes': 10,
                'input_channels': 3,
                'input_size': (32, 32),
                'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
            },
            'cifar100': {
                'num_classes': 100,
                'input_channels': 3,
                'input_size': (32, 32),
                'classes': None  # Too many to list
            },
            'mnist': {
                'num_classes': 10,
                'input_channels': 1,
                'input_size': (28, 28),
                'classes': [str(i) for i in range(10)]
            }
        }
        
        return info.get(dataset_name.lower(), {})

def create_image_datasets(dataset_name: str = 'cifar10', **kwargs):
    """
    Convenience function to create image datasets
    
    Args:
        dataset_name: Name of the dataset
        **kwargs: Additional arguments
        
    Returns:
        Dataset instance
    """
    return ImageDatasetFactory.create_dataset(dataset_name, **kwargs)