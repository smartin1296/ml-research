# Core API Reference

Documentation for shared utilities and core functionality across all algorithms.

## Device Utilities

### `algorithms.*.core.device_utils`

Functions for cross-platform hardware acceleration and optimization.

```python
from algorithms.rnn.core.device_utils import get_best_device, should_use_mixed_precision

# Get optimal device for training
device = get_best_device()
# Returns: 'mps' (Apple Silicon), 'cuda:0' (NVIDIA), or 'cpu' (fallback)

# Check if mixed precision should be used
use_amp = should_use_mixed_precision(device)
# Returns: True for NVIDIA GPUs with Tensor Cores, False for MPS/CPU
```

**Available Functions:**
- `get_best_device()` - Auto-select optimal device (MPS > CUDA > CPU)
- `should_use_mixed_precision(device)` - Check if AMP should be enabled
- `get_device_info()` - Detailed hardware information
- `benchmark_device()` - Performance benchmark for device selection

## Benchmarking Utilities  

### `utils.benchmarking`

Comprehensive performance benchmarking and statistical analysis.

```python
from utils.benchmarking import ModelBenchmark

# Create benchmark suite
benchmark = ModelBenchmark()

# Add models to compare
benchmark.add_model('LSTM', lstm_model, train_loader, val_loader)
benchmark.add_model('GRU', gru_model, train_loader, val_loader)

# Run comparison
results = benchmark.run_comparison()

# Results include:
# - Training time
# - Validation accuracy
# - Memory usage
# - Statistical significance tests
```

## Statistics and Logging

### `utils.stats`

Experiment tracking and statistical analysis utilities.

```python
from utils.stats import ExperimentLogger, ModelStatistics

# Track experiments
logger = ExperimentLogger('experiment_name')
logger.log_hyperparameters({'lr': 0.001, 'batch_size': 32})
logger.log_metrics({'train_loss': 1.23, 'val_acc': 0.85})

# Statistical analysis  
stats = ModelStatistics(results_list)
stats.compute_confidence_intervals()
stats.test_significance()
```

## Data Utilities

### `utils.data_utils`

Dataset handling and preprocessing utilities.

```python
from utils.data_utils import TextDatasetLoader, ImageDatasetLoader

# Text data loading
text_loader = TextDatasetLoader('data/raw/text/shakespeare.txt')
train_data, val_data = text_loader.create_datasets(split_ratio=0.9)

# Image data loading  
image_loader = ImageDatasetLoader('cifar10')
train_loader, val_loader = image_loader.get_loaders(batch_size=128)
```

## Training Framework

### Core Training Components

All algorithms share common training infrastructure:

```python
# Example from RNN implementation
from algorithms.rnn.core.trainer import RNNTrainer

trainer = RNNTrainer(
    model=model,
    device=device,
    mixed_precision=use_amp
)

results = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    lr=0.001
)
```

**Common Features:**
- **Checkpointing**: Automatic best model saving
- **Early Stopping**: Configurable patience and criteria
- **Learning Rate Scheduling**: Multiple scheduler options
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Automatic on supported hardware

## Configuration Management

### Standard Configuration Patterns

```python
# Example configuration structure
config = {
    'model': {
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.1
    },
    'training': {
        'lr': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'scheduler': 'cosine'
    },
    'hardware': {
        'device': 'auto',  # Auto-detect best device
        'mixed_precision': 'auto',  # Enable on compatible hardware
        'num_workers': 4
    }
}
```

## Error Handling

### Common Exception Types

```python
from algorithms.core.exceptions import (
    DeviceNotAvailableError,
    ModelNotTrainedError,
    DatasetNotFoundError
)

try:
    device = get_best_device()
except DeviceNotAvailableError:
    print("No GPU available, falling back to CPU")
    device = 'cpu'
```

## Results Management

### Standard Results Format

All algorithms produce results in consistent format:

```python
{
    "experiment_info": {
        "algorithm": "RNN",
        "timestamp": "2025-08-26T12:34:56",
        "hardware": "Apple M1 Max"
    },
    "model_config": {
        "parameters": 1234567,
        "architecture": "LSTM"
    },
    "training_results": {
        "epochs_trained": 25,
        "best_val_accuracy": 0.856,
        "training_time_seconds": 145.2
    },
    "performance_metrics": {
        "samples_per_second": 5720,
        "memory_usage_mb": 1024
    }
}
```

## Utility Functions

### File Management

```python
from utils.file_utils import (
    save_results,
    load_checkpoint,
    cleanup_temp_files
)

# Save experiment results
save_results(results, "experiments/rnn_20250826_123456/")

# Load model checkpoint
model, optimizer = load_checkpoint("checkpoints/best_model.pt")

# Clean up temporary files
cleanup_temp_files("experiments/")
```

### Reproducibility

```python
from utils.reproducibility import set_random_seeds, get_system_info

# Set seeds for reproducible results
set_random_seeds(42)

# Get system information for documentation
system_info = get_system_info()
# Returns: Python version, PyTorch version, CUDA version, etc.
```

---

**See algorithm-specific API documentation:**
- [RNN API](rnn.md)
- [CNN API](cnn.md) 
- [Transformers API](transformers.md)