import torch
import os
import sys

# Add necessary paths
sys.path.append(os.path.join(os.getcwd(), 'data'))
sys.path.append(os.path.join(os.getcwd(), 'models', 'baseline'))
sys.path.append(os.path.join(os.getcwd(), 'models', 'attention'))

from dataset_utils import CharVocab
from baseline_model import Seq2SeqBaseline
from attention_model import Seq2SeqAttention

# --- Configuration ---
VOCAB_PATH = "data/char_vocab.json"
BASELINE_WEIGHTS_PATH = "models/baseline/baseline_weights.pth"
ATTENTION_WEIGHTS_PATH = "models/attention/attention_weights.pth"

# Hyperparameters (must match training)
EMBED_SIZE = 64
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.0

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    """Loads the vocabulary, baseline model, and attention model."""
    
    # 1. Load Vocabulary
    try:
        vocab = CharVocab.load(VOCAB_PATH)
        vocab_size = vocab.vocab_size
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {VOCAB_PATH}.")
        return None, None, None
        
    # 2. Initialize Models
    baseline_model = Seq2SeqBaseline(
        vocab_size=vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    attention_model = Seq2SeqAttention(
        vocab_size=vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # 3. Load Weights
    try:
        baseline_model.load_state_dict(torch.load(BASELINE_WEIGHTS_PATH, map_location=DEVICE))
        baseline_model.eval()
        print("Baseline model loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: Baseline weights not found at {BASELINE_WEIGHTS_PATH}. Model will be untrained.")
        
    try:
        attention_model.load_state_dict(torch.load(ATTENTION_WEIGHTS_PATH, map_location=DEVICE))
        attention_model.eval()
        print("Attention model loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: Attention weights not found at {ATTENTION_WEIGHTS_PATH}. Model will be untrained.")
        
    return vocab, baseline_model, attention_model

if __name__ == '__main__':
    # Test loading
    vocab, baseline, attention = load_models()
    if vocab and baseline and attention:
        print(f"Vocab size: {vocab.vocab_size}")
        print(f"Baseline model parameters: {sum(p.numel() for p in baseline.parameters())}")
        print(f"Attention model parameters: {sum(p.numel() for p in attention.parameters())}")