"""
Attention-based Decoder for Seq2Seq
Implements Bahdanau Attention Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Bahdanau Attention (Additive Attention)
    Reference: https://arxiv.org/abs/1409.0473
    """
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: [batch_size, dec_hid_dim] - Decoder hidden state
            encoder_outputs: [batch_size, src_len, enc_hid_dim] - All encoder outputs
        
        Returns:
            attention_weights: [batch_size, src_len] - Attention weights (sum=1)
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        # hidden: [batch_size, dec_hid_dim] -> [batch_size, src_len, dec_hid_dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Concatenate hidden and encoder_outputs
        # [batch_size, src_len, enc_hid_dim + dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Calculate attention scores
        # [batch_size, src_len, 1] -> [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        
        # Apply softmax to get attention weights
        return F.softmax(attention, dim=1)


class AttentionDecoder(nn.Module):
    """
    LSTM Decoder with Attention Mechanism
    """
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        
        # Embedding layer
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # LSTM: input = embedding + context vector
        self.rnn = nn.LSTM(
            emb_dim + enc_hid_dim,  # Input: concat(embedding, context)
            dec_hid_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer: hidden + context + embedding -> output
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        """
        Args:
            input: [batch_size, 1] - Current target token
            hidden: [n_layers, batch_size, dec_hid_dim] - Previous hidden state
            cell: [n_layers, batch_size, dec_hid_dim] - Previous cell state
            encoder_outputs: [batch_size, src_len, enc_hid_dim] - All encoder outputs
        
        Returns:
            prediction: [batch_size, output_dim] - Output logits
            hidden: [n_layers, batch_size, dec_hid_dim] - Updated hidden
            cell: [n_layers, batch_size, dec_hid_dim] - Updated cell
            attention_weights: [batch_size, src_len] - Attention weights for visualization
        """
        # input: [batch_size, 1]
        
        # Embedding: [batch_size, 1] -> [batch_size, 1, emb_dim]
        embedded = self.dropout(self.embedding(input))
        
        # Calculate attention using top layer hidden state
        # hidden[-1]: [batch_size, dec_hid_dim]
        attention_weights = self.attention(hidden[-1], encoder_outputs)
        
        # attention_weights: [batch_size, src_len]
        # Apply attention to encoder outputs
        # [batch_size, 1, src_len] x [batch_size, src_len, enc_hid_dim]
        # -> [batch_size, 1, enc_hid_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        # Concatenate embedded input and context
        # [batch_size, 1, emb_dim + enc_hid_dim]
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # LSTM forward
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output: [batch_size, 1, dec_hid_dim]
        
        # Remove seq_len dimension
        embedded = embedded.squeeze(1)  # [batch_size, emb_dim]
        output = output.squeeze(1)      # [batch_size, dec_hid_dim]
        context = context.squeeze(1)    # [batch_size, enc_hid_dim]
        
        # Concatenate output, context, embedded for final prediction
        # [batch_size, dec_hid_dim + enc_hid_dim + emb_dim]
        prediction_input = torch.cat((output, context, embedded), dim=1)
        
        # Final prediction
        prediction = self.fc_out(prediction_input)
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden, cell, attention_weights


class Seq2SeqWithAttention(nn.Module):
    """
    Complete Seq2Seq model with Attention
    """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [batch_size, src_len] - Source sentence
            src_len: [batch_size] - Source lengths
            trg: [batch_size, trg_len] - Target sentence
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: [batch_size, trg_len, output_dim] - Model predictions
            attentions: [batch_size, trg_len, src_len] - Attention weights (for visualization)
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Tensor to store attention weights
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)
        
        # Encoder forward pass
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        # encoder_outputs: [batch_size, src_len, enc_hid_dim]
        
        # First input: <sos> token
        input = trg[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Decode step by step
        for t in range(1, trg_len):
            # Decoder forward pass with attention
            output, hidden, cell, attention = self.decoder(
                input, hidden, cell, encoder_outputs
            )
            
            # Store prediction and attention
            outputs[:, t, :] = output
            attentions[:, t, :] = attention
            
            # Decide next input (teacher forcing or predicted token)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get predicted token
            top1 = output.argmax(1).unsqueeze(1)  # [batch_size, 1]
            
            # Choose next input
            input = trg[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs, attentions


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    """
    Example usage of Attention-based Seq2Seq
    """
    
    # Hyperparameters
    INPUT_DIM = 10000
    OUTPUT_DIM = 10000
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import regular Encoder (from your existing code)
    from encoder import Encoder
    
    # Initialize Encoder
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT)
    
    # Initialize Attention
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    
    # Initialize Attention Decoder
    dec = AttentionDecoder(
        OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, 
        N_LAYERS, DEC_DROPOUT, attn
    )
    
    # Initialize Seq2Seq with Attention
    model = Seq2SeqWithAttention(enc, dec, device).to(device)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # Test forward pass
    src = torch.randint(0, INPUT_DIM, (32, 20)).to(device)  # [batch, src_len]
    src_len = torch.randint(10, 20, (32,)).to(device)
    trg = torch.randint(0, OUTPUT_DIM, (32, 15)).to(device)  # [batch, trg_len]
    
    outputs, attentions = model(src, src_len, trg, teacher_forcing_ratio=0.5)
    
    print(f"Outputs shape: {outputs.shape}")        # [32, 15, 10000]
    print(f"Attentions shape: {attentions.shape}")  # [32, 15, 20]
    print("✅ Attention Decoder working correctly!")
