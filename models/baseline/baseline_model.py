import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq, input_lengths):
        # input_seq: (batch_size, seq_len)
        # input_lengths: (batch_size)
        
        embedded = self.embedding(input_seq) # (batch_size, seq_len, embed_size)
        
        # Pack padded batch of sequences for RNN
        packed = pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Forward pass through GRU
        # output: (batch_size, seq_len, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_size)
        output, hidden = self.rnn(packed)
        
        # Unpack padding (optional, but good practice)
        output, _ = pad_packed_sequence(output, batch_first=True)
        
        # The baseline model only uses the final hidden state
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_step, last_hidden):
        # input_step: (batch_size, 1) - single token
        # last_hidden: (num_layers, batch_size, hidden_size) - context from encoder
        
        embedded = self.embedding(input_step) # (batch_size, 1, embed_size)
        
        # Forward pass through GRU
        # output: (batch_size, 1, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        output, hidden = self.rnn(embedded, last_hidden)
        
        # Apply linear layer and softmax
        # output[:, 0, :] is (batch_size, hidden_size)
        output = self.out(output[:, 0, :]) # (batch_size, vocab_size)
        output = self.softmax(output.unsqueeze(1)) # (batch_size, 1, vocab_size)
        
        return output, hidden

class Seq2SeqBaseline(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        
    def forward(self, input_seq, input_lengths, target_seq, vocab):
        # Encoder forward pass
        encoder_output, encoder_hidden = self.encoder(input_seq, input_lengths)
        
        # Decoder initial state is the final encoder hidden state
        decoder_hidden = encoder_hidden
        
        # Prepare target sequence
        max_target_len = target_seq.size(1)
        batch_size = target_seq.size(0)
        
        # Prepare tensor to store decoder outputs
        decoder_outputs = torch.zeros(batch_size, max_target_len, self.decoder.vocab_size, device=input_seq.device)
        
        # Decoder's first input is the SOS token (which is the first token of target_seq)
        decoder_input = target_seq[:, 0].unsqueeze(1) # (batch_size, 1)
        
        # Teacher forcing: Feed the target as the next input
        for t in range(max_target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Store output
            decoder_outputs[:, t:t+1, :] = decoder_output
            
            # Next input is the current target token
            if t < max_target_len - 1:
                decoder_input = target_seq[:, t+1].unsqueeze(1)
            else:
                break
                
        return decoder_outputs
    
    #def predict(self, input_seq, input_lengths, vocab, max_length=50):
    #     with torch.no_grad():
    #         encoder_output, encoder_hidden = self.encoder(input_seq, input_lengths)
    #         decoder_hidden = encoder_hidden
            
    #         batch_size = input_seq.size(0)
            
    #         # Decoder's first input is the SOS token
    #         decoder_input = torch.tensor([[vocab.sos_idx]] * batch_size, device=input_seq.device)
            
    #         decoded_words = [[] for _ in range(batch_size)]
            
    #         for t in range(max_length):
    #             decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                
    #             # Get the token with the highest probability
    #             topv, topi = decoder_output.topk(1)
                
    #             # topi is (batch_size, 1, 1), squeeze to (batch_size)
    #             next_token = topi.squeeze().squeeze(-1)
                
    #             for i in range(batch_size):
    #                 token_idx = next_token[i].item()
    #                 if token_idx == vocab.eos_idx:
    #                     # Stop decoding for this sequence
    #                     pass # Keep predicting for other sequences in the batch
    #                 elif token_idx != vocab.pad_idx:
    #                     decoded_words[i].append(token_idx)
                
    #             # Set next input to the predicted token
    #             decoder_input = next_token.unsqueeze(1)
                
    #             # Break if all sequences have predicted EOS
    #             if all(vocab.eos_idx in seq for seq in decoded_words):
    #                 break
                    
    #         return decoded_words
    def predict(self, input_seq, input_lengths, vocab, max_length=50):
            with torch.no_grad():
                encoder_output, encoder_hidden = self.encoder(input_seq, input_lengths)
                decoder_hidden = encoder_hidden
                
                batch_size = input_seq.size(0)
                
                # Decoder's first input is the SOS token
                decoder_input = torch.tensor([[vocab.sos_idx]] * batch_size, device=input_seq.device)
                
                decoded_words = [[] for _ in range(batch_size)]
                
                for t in range(max_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    
                    # Get the token with the highest probability
                    topv, topi = decoder_output.topk(1)
                    
                    # topi is (batch_size, 1, 1). 
                    # reshape(-1) ensures we get a 1D tensor of shape (batch_size,) even if batch_size is 1
                    next_token = topi.reshape(-1) 
                    
                    for i in range(batch_size):
                        token_idx = next_token[i].item()
                        if token_idx == vocab.eos_idx:
                            # Stop decoding for this sequence
                            pass # Keep predicting for other sequences in the batch
                        elif token_idx != vocab.pad_idx:
                            decoded_words[i].append(token_idx)
                    
                    # Set next input to the predicted token
                    decoder_input = next_token.unsqueeze(1)
                    
                    # Break if all sequences have predicted EOS
                    if all(vocab.eos_idx in seq for seq in decoded_words):
                        break
                        
                return decoded_words


if __name__ == '__main__':
    # Simple test
    VOCAB_SIZE = 40
    EMBED_SIZE = 16
    HIDDEN_SIZE = 32
    NUM_LAYERS = 1
    
    model = Seq2SeqBaseline(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    
    # Dummy data
    batch_size = 4
    max_len = 10
    dummy_input = torch.randint(0, VOCAB_SIZE, (batch_size, max_len))
    dummy_target = torch.randint(0, VOCAB_SIZE, (batch_size, max_len + 1))
    dummy_lengths = torch.tensor([10, 8, 5, 9])
    
    # Dummy vocab object with required attributes
    class DummyVocab:
        def __init__(self):
            self.sos_idx = 1
            self.eos_idx = 2
            self.pad_idx = 0
            self.vocab_size = VOCAB_SIZE
    
    dummy_vocab = DummyVocab()
    
    output = model(dummy_input, dummy_lengths, dummy_target, dummy_vocab)
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    predictions = model.predict(dummy_input, dummy_lengths, dummy_vocab)
    print(f"Predictions (indices): {predictions}")