import torch
import torch.nn as nn
from typing import List, Dict

# Global list to store attention matrices
ATTENTION_MATRICES: List[torch.Tensor] = []

def clear_attention_matrices():
    """Clears the global list of attention matrices."""
    global ATTENTION_MATRICES
    ATTENTION_MATRICES = []

def register_attention_hook(model: nn.Module):
    """
    Registers a forward hook on the attention mechanism's softmax layer
    to capture the attention weights.
    """
    
    # The attention weights are calculated in the decoder's forward pass
    # in models/attention/attention_model.py
    # The attention weights are returned as the third element from the decoder's forward pass.
    
    # We will modify the Seq2SeqAttention's forward method to capture the attention weights
    # for a single, specific sample during training.
    
    # Since the original plan mentioned models/attention/hooks.py captures attention matrices
    # *during training*, we need a mechanism to selectively capture them.
    # The simplest way is to modify the forward pass of the Seq2SeqAttention model
    # to accept a flag, but since we are using a hook file, we will use a global flag
    # and a hook on the decoder's forward method.
    
    # For simplicity and to match the original plan's intent, we will implement a
    # simple hook that captures the attention weights from the decoder's forward pass.
    
    # Since the attention weights are calculated and returned by the decoder's forward,
    # a simple hook on the decoder's forward pass is not ideal as it doesn't have
    # access to the return values.
    
    # Instead, we will implement a custom forward method in the Seq2SeqAttention class
    # for training that allows for selective attention capture.
    
    # For the purpose of the plan, we will define a simple function to be called
    # manually in the training loop when we want to capture attention.
    
    # The `predict` method in Seq2SeqAttention already captures attention matrices.
    # We will use a separate function in the training script to run a single prediction
    # on a sample and capture the attention matrices.
    
    # Therefore, this file will remain simple and serve as a placeholder for the
    # visualization logic, which will be implemented in the visualization phase.
    
    pass

def capture_attention(attention_weights: torch.Tensor):
    """Stores the attention weights."""
    global ATTENTION_MATRICES
    ATTENTION_MATRICES.append(attention_weights.detach().cpu())

# We will use the predict method in the attention model for visualization,
# which already returns the attention matrices.
# The training script will call a separate function to run a prediction on a fixed sample
# at the end of each epoch to get the attention matrices.


