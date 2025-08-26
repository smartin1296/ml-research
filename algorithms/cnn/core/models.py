import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

class ConvBlock(nn.Module):
    """
    Standard convolutional block with BatchNorm and activation
    
    Features:
    - Proper initialization (He/Kaiming for ReLU, Xavier for other activations)
    - Batch normalization
    - Configurable activation function
    - Optional dropout
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, activation: str = 'relu',
                 batch_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=not batch_norm)
        
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        
        # Activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'silu':
            self.activation = nn.SiLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
            
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        self._init_weights(activation)
    
    def _init_weights(self, activation: str):
        """Initialize weights based on activation function"""
        if activation.lower() in ['relu', 'silu']:
            # He/Kaiming initialization for ReLU-like activations
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        else:
            # Xavier initialization for other activations
            nn.init.xavier_uniform_(self.conv.weight)
            
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        if self.activation is not None:
            x = self.activation(x)
            
        if self.dropout is not None and self.training:
            x = self.dropout(x)
            
        return x

class ResidualBlock(nn.Module):
    """
    Residual block from ResNet paper (He et al., 2016)
    Supports both basic and bottleneck architectures
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, bottleneck: bool = False):
        super().__init__()
        
        self.bottleneck = bottleneck
        
        if bottleneck:
            # Bottleneck: 1x1 -> 3x3 -> 1x1
            mid_channels = out_channels // 4
            self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0)
            self.conv2 = ConvBlock(mid_channels, mid_channels, kernel_size=3, stride=stride)
            self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            # Basic: 3x3 -> 3x3
            self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for residual connections"""
        if self.bottleneck:
            nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(self.bn3.weight)  # Zero-initialize last BN in each residual branch
        else:
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(self.bn2.weight)  # Zero-initialize last BN in each residual branch
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        
        if self.bottleneck:
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.bn3(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class SimpleCNN(nn.Module):
    """
    Simple CNN for baseline comparison
    Architecture: Conv layers -> Global Average Pooling -> Classifier
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3, 
                 base_channels: int = 64, num_layers: int = 4):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.append(ConvBlock(in_channels, out_channels))
            
            # Add max pooling every other layer
            if i % 2 == 1:
                layers.append(nn.MaxPool2d(2, 2))
            
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)
        
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier layer"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNet(nn.Module):
    """
    ResNet implementation based on "Deep Residual Learning for Image Recognition"
    Supports ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
    """
    
    def __init__(self, num_classes: int = 1000, input_channels: int = 3,
                 layers: List[int] = [2, 2, 2, 2], bottleneck: bool = False):
        super().__init__()
        
        self.in_channels = 64
        self.bottleneck = bottleneck
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        final_channels = 512 * (4 if bottleneck else 1)
        self.fc = nn.Linear(final_channels, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a residual layer"""
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * (4 if self.bottleneck else 1):
            final_channels = out_channels * (4 if self.bottleneck else 1)
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, final_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(final_channels),
            )
        
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels * (4 if self.bottleneck else 1), 
                                  stride, downsample, self.bottleneck))
        
        self.in_channels = out_channels * (4 if self.bottleneck else 1)
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels * (4 if self.bottleneck else 1),
                                      bottleneck=self.bottleneck))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize all weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Predefined ResNet architectures
def resnet18(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """ResNet-18"""
    return ResNet(num_classes, input_channels, [2, 2, 2, 2], bottleneck=False)

def resnet34(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """ResNet-34"""
    return ResNet(num_classes, input_channels, [3, 4, 6, 3], bottleneck=False)

def resnet50(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """ResNet-50"""
    return ResNet(num_classes, input_channels, [3, 4, 6, 3], bottleneck=True)

def resnet101(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """ResNet-101"""
    return ResNet(num_classes, input_channels, [3, 4, 23, 3], bottleneck=True)

def resnet152(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """ResNet-152"""
    return ResNet(num_classes, input_channels, [3, 8, 36, 3], bottleneck=True)