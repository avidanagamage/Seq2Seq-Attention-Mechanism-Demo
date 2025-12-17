import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm
import sys
import numpy as np

# Simple path fix - add project root
sys.path.append('../../')

from data.dataset_utils import CharVocab, collate_fn
from attention_model import Seq2SeqAttention

# --- Configuration ---
VOCAB_PATH = "../../data/char_vocab.json"
TRAIN_PATH = "../../data/train_data.pkl"
MODEL_SAVE_PATH = "attention_weights.pth"
ATTENTION_SAMPLES_PATH = "attention_samples.pkl"

# Hyperparameters
EMBED_SIZE = 64
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.0
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 15

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StringDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_epoch(model, dataloader, optimizer, criterion, vocab):
    model.train()
    total_loss = 0
    
    for input_seq, input_lengths, target_seq, _ in dataloader:
        input_seq = input_seq.to(DEVICE)
        input_lengths = input_lengths.to(DEVICE)
        target_seq = target_seq.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(input_seq, input_lengths, target_seq, vocab)
        
        # Target sequence for loss calculation (excluding SOS token)
        target_for_loss = target_seq[:, 1:].contiguous()
        
        # Slice output to match target_for_loss length
        output = output[:, :target_for_loss.size(1), :]
        
        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        target_for_loss = target_for_loss.reshape(-1)
        
        # Mask padded tokens in the loss calculation
        loss = criterion(output, target_for_loss)
        
        # Backward and optimize
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def capture_attention_for_sample(model, vocab, sample_input_str, sample_target_str):
    """
    Runs a prediction on a single sample and captures the attention matrix.
    """
    model.eval()
    
    # Prepare input sequence
    input_indices = vocab.encode(sample_input_str) + [vocab.eos_idx]
    input_tensor = torch.tensor([input_indices], dtype=torch.long, device=DEVICE)
    input_lengths = torch.tensor([len(input_indices)], dtype=torch.long, device=DEVICE)
    
    # Predict
    _, attention_matrices = model.predict(input_tensor, input_lengths, vocab, max_length=len(sample_target_str) + 1)
    
    # attention_matrices is (B, T_out, T_in), B=1
    if attention_matrices is not None:
        return attention_matrices[0] # (T_out, T_in)
    return None

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Vocabulary
    try:
        vocab = CharVocab.load(VOCAB_PATH)
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {VOCAB_PATH}. Please run data/generate_dataset.py first.")
        return
    
    # 2. Load Data
    try:
        with open(TRAIN_PATH, 'rb') as f:
            train_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {TRAIN_PATH}. Please run data/generate_dataset.py first.")
        return
        
    train_dataset = StringDataset(train_data)
    
    # 3. Create DataLoader
    collate = lambda batch: collate_fn(batch, vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    
    # 4. Initialize Model, Loss, and Optimizer
    model = Seq2SeqAttention(
        vocab_size=vocab.vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    criterion = nn.NLLLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    
    # 5. Select a sample for attention visualization
    # Use a fixed, medium-length sample for consistent visualization
    sample_input_str = "attention"
    sample_target_str = sample_input_str[::-1]
    
    # 6. Training Loop
    print("Starting training...")
    best_loss = float('inf')
    attention_samples = {
        "input": sample_input_str,
        "target": sample_target_str,
        "epochs": []
    }
    
    for epoch in range(1, N_EPOCHS + 1):
        loss = train_epoch(model, train_dataloader, optimizer, criterion, vocab)
        
        print(f"Epoch {epoch}/{N_EPOCHS} | Loss: {loss:.4f}")
        
        # Capture attention matrix for the sample
        attention_matrix = capture_attention_for_sample(model, vocab, sample_input_str, sample_target_str)
        if attention_matrix is not None:
            attention_samples["epochs"].append({
                "epoch": epoch,
                "attention_matrix": attention_matrix
            })
            
        # Save best model
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")
            
        # Save attention samples periodically
        if epoch % 2 == 0 or epoch == N_EPOCHS:
            with open(ATTENTION_SAMPLES_PATH, 'wb') as f:
                pickle.dump(attention_samples, f)
            print(f"Attention samples saved to {ATTENTION_SAMPLES_PATH}")

    print("Training complete.")

if __name__ == '__main__':
    
    
    main()