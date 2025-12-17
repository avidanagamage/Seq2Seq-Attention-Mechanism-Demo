import json
import torch
import numpy as np
from typing import List, Tuple, Dict

# Special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'

class CharVocab:
    """
    Handles character-to-index and index-to-character mapping.
    """
    def __init__(self, chars: List[str]):
        self.special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        self.chars = sorted(list(set(chars)))
        self.all_chars = self.special_tokens + self.chars
        
        self.char_to_index = {char: i for i, char in enumerate(self.all_chars)}
        self.index_to_char = {i: char for char, i in self.char_to_index.items()}
        
        self.vocab_size = len(self.all_chars)
        self.pad_idx = self.char_to_index[PAD_TOKEN]
        self.sos_idx = self.char_to_index[SOS_TOKEN]
        self.eos_idx = self.char_to_index[EOS_TOKEN]

    def encode(self, s: str) -> List[int]:
        """Encodes a string into a list of indices."""
        return [self.char_to_index.get(c, self.pad_idx) for c in s]

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Decodes a list of indices back into a string."""
        chars = [self.index_to_char.get(i, '') for i in indices]
        if remove_special:
            chars = [c for c in chars if c not in self.special_tokens]
        return "".join(chars)

    def save(self, path: str):
        """Saves the vocabulary to a JSON file."""
        with open(path, 'w') as f:
            json.dump({
                "char_to_index": self.char_to_index,
                "index_to_char": self.index_to_char,
                "chars": self.chars
            }, f, indent=4)

    @classmethod
    def load(cls, path: str) -> 'CharVocab':
        """Loads the vocabulary from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct the CharVocab object
        instance = cls(data["chars"])
        return instance

def pad_sequence(sequences: List[List[int]], pad_idx: int, max_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads sequences to the maximum length in the batch or a specified max_len.
    Returns padded tensor and original lengths.
    """
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    if max_len is None:
        max_len = lengths.max().item()
    
    padded_sequences = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        # Ensure sequence length does not exceed max_len
        length = min(len(seq), max_len)
        padded_sequences[i, :length] = torch.tensor(seq[:length], dtype=torch.long)
        
    return padded_sequences, lengths

def collate_fn(batch: List[Tuple[str, str]], vocab: CharVocab) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for PyTorch DataLoader.
    Takes a batch of (input_string, target_string) and converts to padded tensors.
    """
    input_strings, target_strings = zip(*batch)
    
    # Encode and add SOS/EOS tokens
    # Input: string -> indices + EOS
    # Target: indices + SOS -> indices + EOS
    
    input_sequences = [vocab.encode(s) + [vocab.eos_idx] for s in input_strings]
    target_sequences = [[vocab.sos_idx] + vocab.encode(s) + [vocab.eos_idx] for s in target_strings]
    
    # Pad sequences
    input_padded, input_lengths = pad_sequence(input_sequences, vocab.pad_idx)
    target_padded, target_lengths = pad_sequence(target_sequences, vocab.pad_idx)
    
    return input_padded, input_lengths, target_padded, target_lengths

if __name__ == '__main__':
    # Simple test
    chars = "abcde"
    vocab = CharVocab(chars)
    
    test_string = "cab"
    encoded = vocab.encode(test_string)
    print(f"Test string: {test_string}")
    print(f"Encoded: {encoded}")
    
    decoded = vocab.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Test padding
    sequences = [
        vocab.encode("a") + [vocab.eos_idx],
        vocab.encode("abc") + [vocab.eos_idx],
        vocab.encode("abcde") + [vocab.eos_idx]
    ]
    padded, lengths = pad_sequence(sequences, vocab.pad_idx)
    print(f"\nSequences: {sequences}")
    print(f"Padded:\n{padded}")
    print(f"Lengths: {lengths}")
    
    # Test save/load
    vocab.save("data/test_vocab.json")
    loaded_vocab = CharVocab.load("data/test_vocab.json")
    print(f"\nLoaded vocab size: {loaded_vocab.vocab_size}")
    print(f"Loaded pad index: {loaded_vocab.pad_idx}")
    
    # Clean up test file
    import os
    os.remove("data/test_vocab.json")