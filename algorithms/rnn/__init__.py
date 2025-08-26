"""
RNN Module - State-of-the-Art Recurrent Neural Network Implementations

This module provides comprehensive SOTA RNN implementations including:
- LSTM with proper initialization and layer normalization
- GRU for efficient training
- Vanilla RNN for baseline comparison
- Bidirectional variants
- Modern training framework with mixed precision
- Comprehensive benchmarking and evaluation tools
"""

from .models import LSTM, GRU, VanillaRNN, LSTMCell
from .trainer import RNNTrainer, create_optimizer, create_scheduler
from .dataset import (
    CharacterTokenizer, 
    TextSequenceDataset, 
    TextDataLoader,
    create_sample_dataset,
    SequenceGenerator
)

__all__ = [
    # Models
    'LSTM', 'GRU', 'VanillaRNN', 'LSTMCell',
    
    # Training
    'RNNTrainer', 'create_optimizer', 'create_scheduler',
    
    # Data handling
    'CharacterTokenizer', 'TextSequenceDataset', 'TextDataLoader',
    'create_sample_dataset', 'SequenceGenerator'
]