import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import sys
from tqdm import tqdm

# Simple path fix - add project root
sys.path.append('../../')

from data.dataset_utils import CharVocab, collate_fn
from attention_model import Seq2SeqAttention

# --- Configuration ---
VOCAB_PATH = "../../data/char_vocab.json"
TEST_PATH = "../../data/test_data.pkl"
MODEL_LOAD_PATH = "attention_weights.pth"

# Hyperparameters (must match training)
EMBED_SIZE = 64
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.0
BATCH_SIZE = 64
EVAL_LENGTHS = [10, 20, 30, 40, 50]

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StringDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate(model, dataloader, vocab):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for input_seq, input_lengths, target_seq, _ in tqdm(dataloader, desc="Evaluating"):
            input_seq = input_seq.to(DEVICE)
            input_lengths = input_lengths.to(DEVICE)
            
            # Target sequence for comparison (excluding SOS token)
            # target_seq is (B, T+1) -> [SOS, c1, c2, ..., EOS]
            # target_for_comparison is (B, T) -> [c1, c2, ..., EOS]
            target_for_comparison = target_seq[:, 1:].cpu().numpy()
            
            # Predict returns a list of lists of token indices and attention matrices
            predicted_indices_list, _ = model.predict(input_seq, input_lengths, vocab, max_length=target_for_comparison.shape[1])
            
            batch_size = input_seq.size(0)
            
            for i in range(batch_size):
                total_samples += 1
                
                # Convert predicted indices to a string
                predicted_str = vocab.decode(predicted_indices_list[i], remove_special=True)
                
                # Convert target indices to a string
                # Find the index of EOS in the target
                target_indices = target_for_comparison[i]
                eos_idx_in_target = next((j for j, x in enumerate(target_indices) if x == vocab.eos_idx), len(target_indices))
                target_str = vocab.decode(target_indices[:eos_idx_in_target], remove_special=True)
                
                if predicted_str == target_str:
                    total_correct += 1
                    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Vocabulary
    try:
        vocab = CharVocab.load(VOCAB_PATH)
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {VOCAB_PATH}.")
        return
    
    # 2. Load Model
    model = Seq2SeqAttention(
        vocab_size=vocab.vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
        print(f"Model loaded from {MODEL_LOAD_PATH}")
    except FileNotFoundError:
        print(f"Error: Model weights not found at {MODEL_LOAD_PATH}. Please train the model first.")
        return
        
    # 3. Load Test Data
    try:
        with open(TEST_PATH, 'rb') as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {TEST_PATH}.")
        return
        
    # 4. Group data by length for evaluation
    data_by_length = {length: [] for length in EVAL_LENGTHS}
    for input_str, target_str in test_data:
        length = len(input_str)
        if length in data_by_length:
            data_by_length[length].append((input_str, target_str))
            
    results = {}
    
    # 5. Evaluate for each length
    print("Starting evaluation...")
    for length in EVAL_LENGTHS:
        subset = data_by_length[length]
        if not subset:
            print(f"Skipping length {length}: No data found.")
            results[length] = 0.0
            continue
            
        dataset = StringDataset(subset)
        collate = lambda batch: collate_fn(batch, vocab)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        
        accuracy = evaluate(model, dataloader, vocab)
        results[length] = accuracy
        print(f"Length {length} Accuracy: {accuracy:.4f} ({len(subset)} samples)")

    # 6. Print final results
    print("\n--- Attention Model Evaluation Results ---")
    for length, acc in results.items():
        print(f"Length {length}: {acc:.4f}")
    print("---------------------------------------")

if __name__ == '__main__':
    
    
    main()