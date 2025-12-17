import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # We use a bidirectional GRU for the encoder
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=True)

    def forward(self, input_seq, input_lengths):
        # input_seq: (batch_size, seq_len)
        
        embedded = self.embedding(input_seq) # (batch_size, seq_len, embed_size)
        
        # Pack padded batch of sequences for RNN
        packed = pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Forward pass through GRU
        # output: (batch_size, seq_len, hidden_size * 2)
        # hidden: (num_layers * 2, batch_size, hidden_size)
        output, hidden = self.rnn(packed)
        
        # Unpack padding
        output, _ = pad_packed_sequence(output, batch_first=True)
        
        # Sum the forward and backward hidden states for the initial decoder state
        # hidden is (2, B, H) for num_layers=1
        # We take the final layer's hidden state and sum the forward/backward
        # For a single layer, it's hidden[0] + hidden[1]
        # For multiple layers, we take the final layer (hidden[-2:]), sum them up, and use as initial state
        
        # For simplicity and to match the baseline's single-direction decoder, 
        # we will concatenate the final forward and backward hidden states
        # and pass it through a linear layer to match the decoder's hidden size.
        
        # hidden is (num_layers * 2, batch_size, hidden_size)
        # We take the last layer's hidden state: hidden[-2:] -> (2, B, H)
        final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) # (B, 2*H)
        
        # We will handle the initial decoder state in the main Seq2Seq class
        
        return output, hidden

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    E = V^T * tanh(W_h * H + W_s * S)
    """
    def __init__(self, hidden_size):
        super().__init__()
        # W_h * H
        self.W_h = nn.Linear(hidden_size * 2, hidden_size, bias=False) # Encoder output is 2*H
        # W_s * S
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False) # Decoder hidden is H
        # V^T
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (1, B, H) -> (B, H) for linear layer
        # encoder_outputs: (B, T, 2*H)
        
        # 1. Calculate W_s * S
        # (B, H) -> (B, H) -> (B, 1, H)
        decoder_hidden_squeezed = decoder_hidden.squeeze(0)
        s_term = self.W_s(decoder_hidden_squeezed).unsqueeze(1) # (B, 1, H)
        
        # 2. Calculate W_h * H
        # (B, T, 2*H) -> (B, T, H)
        h_term = self.W_h(encoder_outputs) # (B, T, H)
        
        # 3. Calculate E = V^T * tanh(W_h * H + W_s * S)
        # (B, T, H) + (B, 1, H) -> (B, T, H)
        energy = torch.tanh(h_term + s_term)
        
        # (B, T, H) -> (B, T, 1)
        attention_scores = self.V(energy)
        
        # 4. Calculate attention weights (alpha)
        # (B, T, 1) -> (B, T)
        attention_weights = F.softmax(attention_scores, dim=1).squeeze(2) # (B, T)
        
        # 5. Calculate context vector (C)
        # (B, 1, T) * (B, T, 2*H) -> (B, 1, 2*H)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs) # (B, 1, 2*H)
        
        return context_vector, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # GRU input is concatenation of embedded token (E) and context vector (2*H)
        self.rnn = nn.GRU(embed_size + hidden_size * 2, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.attention = BahdanauAttention(hidden_size)
        
        # Output layer takes concatenation of context vector and decoder hidden state
        self.out = nn.Linear(hidden_size * 2 + hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # input_step: (batch_size, 1) - single token
        # last_hidden: (num_layers, batch_size, hidden_size) - previous decoder hidden state
        # encoder_outputs: (batch_size, seq_len, 2*hidden_size) - all encoder outputs
        
        # 1. Calculate Attention
        # We use the hidden state of the last layer of the decoder GRU
        # last_hidden[-1, :, :].unsqueeze(0) is (1, B, H)
        context_vector, attention_weights = self.attention(last_hidden[-1, :, :].unsqueeze(0), encoder_outputs)
        
        # 2. Embed input
        embedded = self.embedding(input_step) # (B, 1, E)
        
        # 3. Concatenate embedded input and context vector
        # (B, 1, E) + (B, 1, 2*H) -> (B, 1, E + 2*H)
        rnn_input = torch.cat((embedded, context_vector), dim=2)
        
        # 4. Forward pass through GRU
        # output: (B, 1, H)
        # hidden: (num_layers, B, H)
        output, hidden = self.rnn(rnn_input, last_hidden)
        
        # 5. Calculate output probability
        # Concatenate output (H) and context vector (2*H)
        # output[:, 0, :] is (B, H)
        # context_vector.squeeze(1) is (B, 2*H)
        output_concat = torch.cat((output[:, 0, :], context_vector.squeeze(1)), dim=1) # (B, 3*H)
        
        # (B, 3*H) -> (B, V)
        output = self.out(output_concat)
        output = self.softmax(output.unsqueeze(1)) # (B, 1, V)
        
        return output, hidden, attention_weights

class Seq2SeqAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Linear layer to transform the concatenated bidirectional encoder hidden state
        # to the unidirectional decoder hidden state size
        self.hidden_transform = nn.Linear(hidden_size * 2, hidden_size * num_layers)

    def _init_decoder_hidden(self, encoder_hidden):
        # encoder_hidden is (num_layers * 2, B, H)
        # We take the final layer's hidden state (hidden[-2:]), concatenate them, 
        # and transform to (num_layers, B, H)
        
        # 1. Take the final layer's hidden state (forward and backward)
        # (2, B, H)
        final_layer_hidden = encoder_hidden[-2:, :, :]
        
        # 2. Concatenate them: (B, 2*H)
        concatenated_hidden = torch.cat((final_layer_hidden[0], final_layer_hidden[1]), dim=1)
        
        # 3. Transform to (B, num_layers * H)
        transformed_hidden = self.hidden_transform(concatenated_hidden)
        
        # 4. Reshape to (num_layers, B, H)
        # We assume num_layers is 1 for simplicity in this project, but this is more general
        decoder_hidden = transformed_hidden.view(self.num_layers, transformed_hidden.size(0), self.hidden_size)
        
        return decoder_hidden

    def forward(self, input_seq, input_lengths, target_seq, vocab):
        # Encoder forward pass
        # encoder_outputs: (B, T, 2*H)
        # encoder_hidden: (num_layers * 2, B, H)
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths)
        
        # Initialize decoder hidden state
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)
        
        # Prepare target sequence
        max_target_len = target_seq.size(1)
        batch_size = target_seq.size(0)
        
        # Prepare tensor to store decoder outputs
        decoder_outputs = torch.zeros(batch_size, max_target_len, self.decoder.vocab_size, device=input_seq.device)
        
        # Decoder's first input is the SOS token
        decoder_input = target_seq[:, 0].unsqueeze(1) # (batch_size, 1)
        
        # Teacher forcing
        for t in range(max_target_len):
            # decoder_output: (B, 1, V)
            # decoder_hidden: (num_layers, B, H)
            # attention_weights: (B, T)
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # Store output
            decoder_outputs[:, t:t+1, :] = decoder_output
            
            # Next input is the current target token
            if t < max_target_len - 1:
                decoder_input = target_seq[:, t+1].unsqueeze(1)
            else:
                break
                
        return decoder_outputs
        
    def predict(self, input_seq, input_lengths, vocab, max_length=50):
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths)
            decoder_hidden = self._init_decoder_hidden(encoder_hidden)
            
            batch_size = input_seq.size(0)
            
            # Decoder's first input is the SOS token
            decoder_input = torch.tensor([[vocab.sos_idx]] * batch_size, device=input_seq.device)
            
            decoded_words = [[] for _ in range(batch_size)]
            attention_matrices = [] # To store attention weights for visualization
            
            for t in range(max_length):
                decoder_output, decoder_hidden, attention_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                
                # Store attention weights for this time step
                attention_matrices.append(attention_weights.cpu().numpy())
                
                # Get the token with the highest probability
                topv, topi = decoder_output.topk(1)
                
                # topi is (batch_size, 1, 1)
                next_token = topi.squeeze(-1).squeeze(-1) # (batch_size)
                
                all_eos = True
                for i in range(batch_size):
                    token_idx = next_token[i].item()
                    if token_idx == vocab.eos_idx:
                        # Stop decoding for this sequence
                        pass
                    elif token_idx != vocab.pad_idx:
                        decoded_words[i].append(token_idx)
                        all_eos = False
                
                # Set next input to the predicted token
                decoder_input = next_token.unsqueeze(1)
                
                # Break if all sequences have predicted EOS
                if all_eos and t > 0:
                    break
                    
            # attention_matrices is a list of (B, T_in) arrays
            # We want to stack them to get (T_out, B, T_in)
            # Then transpose to get (B, T_out, T_in)
            if attention_matrices:
                attention_matrices = torch.tensor(attention_matrices).permute(1, 0, 2).numpy()
            else:
                attention_matrices = None
                    
            return decoded_words, attention_matrices

if __name__ == '__main__':
    # Simple test
    VOCAB_SIZE = 40
    EMBED_SIZE = 64
    HIDDEN_SIZE = 256
    NUM_LAYERS = 1
    
    model = Seq2SeqAttention(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    
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
    predictions, attention_matrices = model.predict(dummy_input, dummy_lengths, dummy_vocab)
    print(f"Predictions (indices): {predictions}")
    print(f"Attention matrix shape: {attention_matrices.shape}") # (B, T_out, T_in)