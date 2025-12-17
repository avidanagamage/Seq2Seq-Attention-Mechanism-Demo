import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add data directory to path to import dataset_utils
sys.path.append(os.path.join(os.getcwd(), 'data'))
from dataset_utils import CharVocab

# --- Configuration ---
ATTENTION_SAMPLES_PATH = "models/attention/attention_samples.pkl"
VOCAB_PATH = "data/char_vocab.json"
HEATMAPS_DIR = "visualization/heatmaps"

def load_data():
    """Loads attention samples and vocabulary."""
    try:
        with open(ATTENTION_SAMPLES_PATH, 'rb') as f:
            attention_samples = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Attention samples file not found at {ATTENTION_SAMPLES_PATH}. Please run train_attention.py first.")
        return None, None
        
    try:
        vocab = CharVocab.load(VOCAB_PATH)
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {VOCAB_PATH}.")
        return None, None
        
    return attention_samples, vocab

def plot_heatmap(matrix: np.ndarray, input_str: str, target_str: str, epoch: int):
    """Generates and saves a heatmap for a single attention matrix."""
    
    # matrix shape is (T_out, T_in)
    # T_out is the length of the predicted output sequence
    # T_in is the length of the input sequence (including EOS)
    
    # Get the actual input and output tokens for labels
    input_tokens = list(input_str) + ['<EOS>']
    # The decoder predicts the next token, so the output tokens are the target string
    # plus the EOS token, but we only plot up to the length of the matrix.
    output_tokens = list(target_str)
    
    # The matrix size might be smaller than the full target sequence if prediction stopped early
    matrix = matrix[:len(output_tokens), :len(input_tokens)]
    
    # Create the plot
    plt.figure(figsize=(len(input_tokens) * 0.8, len(output_tokens) * 0.8))
    
    # Use 'viridis' or 'magma' for a good contrast heatmap
    sns.heatmap(matrix, 
                xticklabels=input_tokens, 
                yticklabels=output_tokens, 
                cmap="viridis", 
                linewidths=0.5, 
                linecolor='lightgray',
                cbar=True)
    
    plt.xlabel("Input Sequence (Encoder)")
    plt.ylabel("Output Sequence (Decoder)")
    plt.title(f"Attention Heatmap - Epoch {epoch}")
    
    # Save the plot
    filename = f"attention_heatmap_epoch_{epoch}.png"
    filepath = os.path.join(HEATMAPS_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {filepath}")

def main():
    # Change directory to project root for correct path resolution
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("../")
    
    attention_samples, vocab = load_data()
    
    if attention_samples is None:
        return
        
    input_str = attention_samples["input"]
    target_str = attention_samples["target"]
    
    print(f"Generating heatmaps for input: '{input_str}' (reversed: '{target_str}')")
    
    # Ensure the output directory exists
    os.makedirs(HEATMAPS_DIR, exist_ok=True)
    
    for sample in attention_samples["epochs"]:
        epoch = sample["epoch"]
        matrix = sample["attention_matrix"]
        
        # The matrix is (T_out, T_in)
        # T_in includes the EOS token
        # T_out is the length of the predicted sequence
        
        plot_heatmap(matrix, input_str, target_str, epoch)

if __name__ == '__main__':
    main()