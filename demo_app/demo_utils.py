import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Add data directory to path to import dataset_utils
sys.path.append(os.path.join(os.getcwd(), 'data'))
from dataset_utils import CharVocab

# --- Configuration ---
VOCAB_PATH = "data/char_vocab.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Vocab object
try:
    VOCAB = CharVocab.load(VOCAB_PATH)
except FileNotFoundError:
    VOCAB = None
    print(f"Warning: Vocab file not found at {VOCAB_PATH}. Demo will not work.")

def preprocess_input(input_str: str) -> tuple:
    """Encodes and prepares the input string for the models."""
    if VOCAB is None:
        return None, None
        
    # 1. Encode and add EOS token
    input_indices = VOCAB.encode(input_str) + [VOCAB.eos_idx]
    
    # 2. Convert to tensor
    input_tensor = torch.tensor([input_indices], dtype=torch.long, device=DEVICE)
    input_lengths = torch.tensor([len(input_indices)], dtype=torch.long, device=DEVICE)
    
    return input_tensor, input_lengths

def postprocess_output(predicted_indices: list) -> str:
    """Decodes the predicted indices back into a string."""
    if VOCAB is None:
        return "Error: Vocab not loaded."
        
    # predicted_indices is a list of lists (batch_size=1)
    if not predicted_indices or not predicted_indices[0]:
        return ""
        
    return VOCAB.decode(predicted_indices[0], remove_special=True)

def generate_attention_heatmap(matrix: np.ndarray, input_str: str, target_str: str) -> BytesIO:
    """
    Generates a Matplotlib heatmap for the attention matrix and returns it as a BytesIO object.
    """
    if matrix is None:
        return None
        
    # matrix shape is (T_out, T_in)
    
    # Get the actual input and output tokens for labels
    input_tokens = list(input_str) + ['<EOS>']
    output_tokens = list(target_str)
    
    # Ensure matrix dimensions match the labels
    T_out, T_in = matrix.shape
    
    # Slice the labels to match the matrix size
    input_tokens = input_tokens[:T_in]
    output_tokens = output_tokens[:T_out]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(len(input_tokens) * 0.8, len(output_tokens) * 0.8))
    
    sns.heatmap(matrix, 
                xticklabels=input_tokens, 
                yticklabels=output_tokens, 
                cmap="viridis", 
                linewidths=0.5, 
                linecolor='lightgray',
                cbar=True,
                ax=ax)
    
    ax.set_xlabel("Input Sequence (Encoder)")
    ax.set_ylabel("Output Sequence (Decoder)")
    ax.set_title("Live Attention Heatmap")
    
    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf

def get_reversed_string(input_str: str) -> str:
    """Simple utility to get the expected output."""
    return input_str[::-1]

def get_vocab() -> CharVocab:
    """Returns the global CharVocab object."""
    return VOCAB

if __name__ == '__main__':
    # Simple test
    test_str = "hello"
    rev_str = get_reversed_string(test_str)
    print(f"Input: {test_str}, Reversed: {rev_str}")
    
    input_tensor, input_lengths = preprocess_input(test_str)
    print(f"Input Tensor Shape: {input_tensor.shape}")
    
    # Dummy attention matrix for testing heatmap generation
    dummy_matrix = np.random.rand(len(rev_str), len(test_str) + 1)
    heatmap_buf = generate_attention_heatmap(dummy_matrix, test_str, rev_str)
    print(f"Heatmap buffer size: {len(heatmap_buf.getvalue())} bytes")