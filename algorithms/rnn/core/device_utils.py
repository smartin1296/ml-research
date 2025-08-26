"""
Device utilities for cross-platform GPU support
Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback
"""

import torch

def get_best_device(prefer_device: str = 'auto') -> torch.device:
    """
    Get the best available device for PyTorch training
    
    Args:
        prefer_device: 'auto', 'mps', 'cuda', or 'cpu'
        
    Returns:
        torch.device: Best available device
    """
    
    if prefer_device == 'cpu':
        return torch.device('cpu')
    elif prefer_device == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            raise RuntimeError("CUDA requested but not available")
    elif prefer_device == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            raise RuntimeError("MPS requested but not available")
    else:  # auto
        # Priority: MPS > CUDA > CPU
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

def get_device_info() -> dict:
    """Get information about available devices"""
    info = {
        'cpu_available': True,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'mps_built': torch.backends.mps.is_built(),
        'best_device': str(get_best_device()),
        'pytorch_version': torch.__version__
    }
    
    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_capability'] = torch.cuda.get_device_capability(0)
    
    return info

def should_use_mixed_precision(device: torch.device) -> bool:
    """
    Determine if mixed precision should be used on the given device
    
    Args:
        device: Target device
        
    Returns:
        bool: Whether to use mixed precision
    """
    # Mixed precision works well on:
    # - Modern NVIDIA GPUs (Tensor Cores)
    # - May not work reliably on MPS yet
    
    if device.type == 'cuda':
        # Check if GPU supports Tensor Cores (compute capability >= 7.0)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return major >= 7  # Tensor Core support
    elif device.type == 'mps':
        # MPS mixed precision support is still experimental
        # Disable for now to avoid issues
        return False
    
    return False

def print_device_info():
    """Print detailed device information"""
    info = get_device_info()
    
    print("🔧 Device Information")
    print("=" * 40)
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"CPU Available: ✅")
    print(f"CUDA Available: {'✅' if info['cuda_available'] else '❌'}")
    print(f"MPS Available: {'✅' if info['mps_available'] else '❌'}")
    print(f"MPS Built: {'✅' if info['mps_built'] else '❌'}")
    print(f"Best Device: {info['best_device']}")
    
    if info['cuda_available']:
        print(f"CUDA Device Count: {info['cuda_device_count']}")
        print(f"CUDA Device Name: {info['cuda_device_name']}")
        print(f"CUDA Capability: {info['cuda_capability']}")
    
    best_device = get_best_device()
    print(f"Mixed Precision Recommended: {'✅' if should_use_mixed_precision(best_device) else '❌'}")

if __name__ == '__main__':
    print_device_info()