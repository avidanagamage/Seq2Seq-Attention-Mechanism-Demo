import random
import pickle
import os
import string
from typing import List, Tuple
from dataset_utils import CharVocab

# --- Configuration ---
# string.punctuation adds: !"#$%&'()*+,-./:;<=>?@[\]^_{|}~`
#CHARACTERS = string.ascii_letters + string.digits + string.punctuation + " "
CHARACTERS = string.ascii_lowercase + string.digits + string.punctuation + " "
DATASET_SIZE = 150000
MAX_LENGTH = 50
TRAIN_TEST_SPLIT = 0.9
SHORT_STRING_MAX_LEN = 50
EVAL_LENGTHS = [10, 20, 30, 40, 50]
EVAL_SAMPLES_PER_LENGTH = 1000

# --- File Paths ---
VOCAB_PATH = "char_vocab.json"
TRAIN_PATH = "train_data.pkl"
TEST_PATH = "test_data.pkl"

def generate_random_string(min_len: int, max_len: int) -> str:
    """Generates a random string of length between min_len and max_len."""
    length = random.randint(min_len, max_len)
    return "".join(random.choice(CHARACTERS) for _ in range(length))

def generate_dataset(size: int, max_len: int) -> List[Tuple[str, str]]:
    """Generates a list of (input_string, reversed_string) pairs."""
    data = []
    for _ in range(size):
        # Generate length between 5 and MAX_LENGTH
        input_str = generate_random_string(5, max_len)
        target_str = input_str[::-1]
        data.append((input_str, target_str))
    return data

def split_dataset(data: List[Tuple[str, str]], split_ratio: float) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Splits the dataset into train and test sets."""
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]

def create_eval_test_set(all_test_data: List[Tuple[str, str]], lengths: List[int], samples_per_length: int) -> List[Tuple[str, str]]:
    """
    Creates a test set specifically for evaluation, ensuring a fixed number of samples
    for each specified length.
    """
    eval_set = []
    
    # Group test data by length
    data_by_length = {}
    for input_str, target_str in all_test_data:
        length = len(input_str)
        if length not in data_by_length:
            data_by_length[length] = []
        data_by_length[length].append((input_str, target_str))
        
    # Sample from each required length
    for length in lengths:
        if length in data_by_length:
            # Filter to only include strings of the exact length
            exact_length_data = [d for d in data_by_length[length] if len(d[0]) == length]
            
            if len(exact_length_data) >= samples_per_length:
                # Randomly sample the required number of samples
                eval_set.extend(random.sample(exact_length_data, samples_per_length))
            else:
                print(f"Warning: Only found {len(exact_length_data)} samples of length {length}. Using all of them.")
                eval_set.extend(exact_length_data)
        else:
            print(f"Warning: No samples of length {length} found in the test set.")
            
    return eval_set

def main():
    # 1. Create CharVocab and save it
    print("Creating CharVocab...")
    vocab = CharVocab(CHARACTERS)
    vocab.save(VOCAB_PATH)
    print(f"CharVocab saved to {VOCAB_PATH}. Vocab size: {vocab.vocab_size}")
    
    # 2. Generate and split the full dataset
    print(f"Generating full dataset of size {DATASET_SIZE}...")
    full_data = generate_dataset(DATASET_SIZE, MAX_LENGTH)
    train_data_raw, test_data_raw = split_dataset(full_data, TRAIN_TEST_SPLIT)
    
    # 3. Filter training data for "short strings"
    print(f"Filtering training data for short strings (max length {SHORT_STRING_MAX_LEN})...")
    train_data_short = [
        (i, t) for i, t in train_data_raw 
        if len(i) <= SHORT_STRING_MAX_LEN
    ]
    print(f"Original train size: {len(train_data_raw)}. Short train size: {len(train_data_short)}")
    
    # 4. Create evaluation test set
    print(f"Creating evaluation test set for lengths {EVAL_LENGTHS}...")
    test_data_eval = create_eval_test_set(test_data_raw, EVAL_LENGTHS, EVAL_SAMPLES_PER_LENGTH)
    print(f"Evaluation test size: {len(test_data_eval)}")
    
    # 5. Save datasets
    print(f"Saving train data to {TRAIN_PATH}...")
    with open(TRAIN_PATH, 'wb') as f:
        pickle.dump(train_data_short, f)
        
    print(f"Saving test data to {TEST_PATH}...")
    with open(TEST_PATH, 'wb') as f:
        pickle.dump(test_data_eval, f)
        
    print("Data generation complete.")

if __name__ == '__main__':
    # Need to change directory to data so the relative import works
    # and the files are saved in the correct location
    #os.chdir("data")
    # Temporarily add the current directory to the path for module import
    import sys
    sys.path.append(os.getcwd())
    
    try:
        # The script will now run in the 'data' directory
        main()
    finally:
        # Clean up path and change back
        sys.path.pop()
        os.chdir("..")