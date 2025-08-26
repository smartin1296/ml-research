import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple, Union
import math

class LSTMCell(nn.Module):
    """
    SOTA LSTM Cell implementation with proper initialization and optional layer normalization.
    
    Features:
    - Xavier initialization for weights
    - Forget gate bias initialized to 1 (Jozefowicz et al., 2015)
    - Optional layer normalization (Ba et al., 2016)
    - Gradient clipping friendly
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 layer_norm: bool = False, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        
        # Input-to-hidden weights (i, f, g, o gates)
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        # Hidden-to-hidden weights
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        # Biases
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        
        # Layer normalization
        if layer_norm:
            self.ln_ih = nn.LayerNorm(4 * hidden_size)  # Enable learnable parameters
            self.ln_hh = nn.LayerNorm(4 * hidden_size)  # Enable learnable parameters
            self.ln_c = nn.LayerNorm(hidden_size)       # Enable learnable parameters
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following best practices"""
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        
        # Initialize biases to zero, except forget gate bias = 1
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
        
        # Set forget gate bias to 1 (indices hidden_size:2*hidden_size)
        nn.init.ones_(self.bias_ih[self.hidden_size:2*self.hidden_size])
        nn.init.ones_(self.bias_hh[self.hidden_size:2*self.hidden_size])
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of LSTM cell
        
        Args:
            x: Input tensor (batch_size, input_size)
            hidden: Tuple of (h_0, c_0) or None
            
        Returns:
            (h_t, c_t): New hidden and cell states
        """
        batch_size = x.size(0)
        
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = hidden
        
        # Linear transformations
        gi = F.linear(x, self.weight_ih, self.bias_ih)
        gh = F.linear(h, self.weight_hh, self.bias_hh)
        
        # Apply layer normalization if enabled
        if self.layer_norm:
            gi = self.ln_ih(gi)
            gh = self.ln_hh(gh)
        
        # Split into gates
        i_i, i_f, i_g, i_o = gi.chunk(4, 1)
        h_i, h_f, h_g, h_o = gh.chunk(4, 1)
        
        # Gate computations
        input_gate = torch.sigmoid(i_i + h_i)
        forget_gate = torch.sigmoid(i_f + h_f)
        cell_gate = torch.tanh(i_g + h_g)
        output_gate = torch.sigmoid(i_o + h_o)
        
        # Cell state update
        c_new = forget_gate * c + input_gate * cell_gate
        
        # Apply layer normalization to cell state if enabled
        if self.layer_norm:
            c_norm = self.ln_c(c_new)
        else:
            c_norm = c_new
            
        # Hidden state update
        h_new = output_gate * torch.tanh(c_norm)
        
        # Apply dropout to hidden state
        if self.dropout is not None and self.training:
            h_new = self.dropout(h_new)
            
        return h_new, c_new

class LSTM(nn.Module):
    """
    Multi-layer LSTM with modern features:
    - Bidirectional support
    - Layer normalization
    - Dropout between layers
    - Packed sequence support
    - Proper weight initialization
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bidirectional: bool = False, layer_norm: bool = False,
                 dropout: float = 0.0, batch_first: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        
        # Create LSTM layers
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            
            if bidirectional:
                # Forward and backward cells
                fw_cell = LSTMCell(layer_input_size, hidden_size, layer_norm, 
                                  dropout if layer < num_layers - 1 else 0.0)
                bw_cell = LSTMCell(layer_input_size, hidden_size, layer_norm,
                                  dropout if layer < num_layers - 1 else 0.0)
                self.layers.append(nn.ModuleList([fw_cell, bw_cell]))
            else:
                cell = LSTMCell(layer_input_size, hidden_size, layer_norm,
                              dropout if layer < num_layers - 1 else 0.0)
                self.layers.append(cell)
    
    def forward(self, x: Union[torch.Tensor, PackedSequence], 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[Union[torch.Tensor, PackedSequence], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through multi-layer LSTM
        
        Args:
            x: Input tensor (seq_len, batch_size, input_size) or PackedSequence
            hidden: Initial hidden state tuple (h_0, c_0)
            
        Returns:
            output: All hidden states
            (h_n, c_n): Final hidden and cell states
        """
        is_packed = isinstance(x, PackedSequence)
        
        if is_packed:
            x, batch_sizes = pad_packed_sequence(x, batch_first=self.batch_first)
        
        if not self.batch_first:
            x = x.transpose(0, 1)  # Convert to (batch, seq, features)
        
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            h = torch.zeros(self.num_layers * self.num_directions, batch_size, 
                          self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(self.num_layers * self.num_directions, batch_size,
                          self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = hidden
        
        output = x
        h_n_list = []
        c_n_list = []
        
        for layer_idx, layer in enumerate(self.layers):
            layer_h = []
            layer_c = []
            
            if self.bidirectional:
                fw_cell, bw_cell = layer
                
                # Forward direction
                fw_h = h[layer_idx * 2]
                fw_c = c[layer_idx * 2]
                fw_outputs = []
                
                for t in range(seq_len):
                    fw_h, fw_c = fw_cell(output[:, t], (fw_h, fw_c))
                    fw_outputs.append(fw_h.unsqueeze(1))
                
                fw_output = torch.cat(fw_outputs, dim=1)
                
                # Backward direction
                bw_h = h[layer_idx * 2 + 1]
                bw_c = c[layer_idx * 2 + 1]
                bw_outputs = []
                
                for t in range(seq_len - 1, -1, -1):
                    bw_h, bw_c = bw_cell(output[:, t], (bw_h, bw_c))
                    bw_outputs.append(bw_h.unsqueeze(1))
                
                bw_output = torch.cat(bw_outputs[::-1], dim=1)
                
                # Concatenate forward and backward outputs
                output = torch.cat([fw_output, bw_output], dim=2)
                
                layer_h.extend([fw_h, bw_h])
                layer_c.extend([fw_c, bw_c])
                
            else:
                cell = layer
                layer_output = []
                cell_h = h[layer_idx]
                cell_c = c[layer_idx]
                
                for t in range(seq_len):
                    cell_h, cell_c = cell(output[:, t], (cell_h, cell_c))
                    layer_output.append(cell_h.unsqueeze(1))
                
                output = torch.cat(layer_output, dim=1)
                layer_h.append(cell_h)
                layer_c.append(cell_c)
            
            h_n_list.extend(layer_h)
            c_n_list.extend(layer_c)
        
        h_n = torch.stack(h_n_list, dim=0)
        c_n = torch.stack(c_n_list, dim=0)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        if is_packed:
            output = pack_padded_sequence(output, batch_sizes.cpu(), 
                                        batch_first=self.batch_first, enforce_sorted=False)
        
        return output, (h_n, c_n)

class GRU(nn.Module):
    """
    Gated Recurrent Unit implementation for comparison with LSTM
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.0, batch_first: bool = True):
        super().__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         bidirectional=bidirectional, dropout=dropout,
                         batch_first=batch_first)
        
        # Apply proper initialization
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: Union[torch.Tensor, PackedSequence], 
                hidden: Optional[torch.Tensor] = None):
        return self.gru(x, hidden)

class VanillaRNN(nn.Module):
    """
    Simple RNN for baseline comparison
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', dropout: float = 0.0, batch_first: bool = True):
        super().__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                         nonlinearity=nonlinearity, dropout=dropout,
                         batch_first=batch_first)
        
        # Apply proper initialization
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: Union[torch.Tensor, PackedSequence], 
                hidden: Optional[torch.Tensor] = None):
        return self.rnn(x, hidden)