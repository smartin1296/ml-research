import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from typing import Tuple, List, Dict, Optional, Union
import os

class ImageDataset(Dataset):
    """Generic image dataset for classification tasks"""
    
    def __init__(self, root_dir: str, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx or {}
        
        self._load_samples()
    
    def _load_samples(self):
        """Load image samples from directory structure"""
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
            
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class TextDataset(Dataset):
    """Generic text dataset for classification or language modeling"""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None,
                 tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            if self.labels is not None:
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }
            else:
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
        else:
            # Simple character or word-level tokenization
            tokens = torch.tensor([ord(c) for c in text[:self.max_length]], dtype=torch.long)
            
            if self.labels is not None:
                return tokens, torch.tensor(self.labels[idx], dtype=torch.long)
            else:
                return tokens

class TabularDataset(Dataset):
    """Generic tabular dataset"""
    
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], 
                 target: Optional[Union[pd.Series, np.ndarray]] = None,
                 feature_cols: Optional[List[str]] = None,
                 target_col: Optional[str] = None):
        
        if isinstance(data, pd.DataFrame):
            if feature_cols:
                self.features = data[feature_cols].values
            elif target_col:
                self.features = data.drop(columns=[target_col]).values
                self.target = data[target_col].values
            else:
                self.features = data.values
        else:
            self.features = data
        
        if target is not None:
            self.target = target.values if isinstance(target, pd.Series) else target
        elif hasattr(self, 'target'):
            pass  # Already set above
        else:
            self.target = None
        
        self.features = torch.tensor(self.features, dtype=torch.float32)
        if self.target is not None:
            self.target = torch.tensor(self.target, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.target is not None:
            return self.features[idx], self.target[idx]
        else:
            return self.features[idx]

class DataManager:
    """Utility class for managing datasets and data loaders"""
    
    @staticmethod
    def get_image_transforms(train: bool = True, img_size: int = 224):
        """Get standard image transforms"""
        if train:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    @staticmethod
    def create_train_val_split(dataset: Dataset, 
                              val_ratio: float = 0.2,
                              random_seed: int = 42) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and validation sets"""
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        
        generator = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        
        return train_dataset, val_dataset
    
    @staticmethod
    def create_data_loaders(train_dataset: Dataset,
                           val_dataset: Dataset,
                           batch_size: int = 32,
                           num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training and validation"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader
    
    @staticmethod
    def save_dataset_info(dataset: Dataset, filepath: str):
        """Save dataset information to file"""
        info = {
            'dataset_type': type(dataset).__name__,
            'length': len(dataset),
            'sample_shape': str(dataset[0][0].shape) if hasattr(dataset[0][0], 'shape') else 'N/A'
        }
        
        if hasattr(dataset, 'class_to_idx'):
            info['classes'] = dataset.class_to_idx
        
        with open(filepath, 'w') as f:
            import json
            json.dump(info, f, indent=2)