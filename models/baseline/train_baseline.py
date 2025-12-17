import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm
import sys



# Simple path fix - add project root
sys.path.append('../../')

from data.dataset_utils import CharVocab, collate_fn

from baseline_model import Seq2SeqBaseline

# --- Configuration ---
VOCAB_PATH = "../../data/char_vocab.json"
TRAIN_PATH = "../../data/train_data.pkl"
MODEL_SAVE_PATH = "baseline_weights.pth"

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
        # The model's forward pass handles the decoding loop and teacher forcing
        output = model(input_seq, input_lengths, target_seq, vocab)
        
        # The target sequence for loss calculation should exclude the SOS token
        # and be flattened for cross-entropy loss.
        # target_seq is (B, T+1) -> [SOS, c1, c2, ..., EOS]
        # target_for_loss is (B, T) -> [c1, c2, ..., EOS]
        target_for_loss = target_seq[:, 1:].contiguous()
        
        # Output is (B, T, V)
        # We need to reshape it to (B*T, V)
        output_dim = output.shape[-1]
        
        # The output from the model is (B, T_out, V). T_out should be T_target - 1.
        # We need to ensure output and target_for_loss have the same sequence length T.
        # The model's forward pass generates T_target - 1 tokens.
        # Let's slice the output to match the target length.
        
        # Output shape is (B, T_target, V) where T_target is max_target_len
        # The actual output sequence length is target_seq.size(1) - 1
        
        # Slice output to match target_for_loss length
        output = output[:, :target_for_loss.size(1), :]
        
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
    # We use a lambda function to pass the vocab object to the collate_fn
    collate = lambda batch: collate_fn(batch, vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    
    # 4. Initialize Model, Loss, and Optimizer
    model = Seq2SeqBaseline(
        vocab_size=vocab.vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Ignore padding index for loss calculation
    criterion = nn.NLLLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    
    # 5. Training Loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(1, N_EPOCHS + 1):
        loss = train_epoch(model, train_dataloader, optimizer, criterion, vocab)
        
        print(f"Epoch {epoch}/{N_EPOCHS} | Loss: {loss:.4f}")
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")

    print("Training complete.")

if __name__ == '__main__':
    
    main()