import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from torch.nn.modules.activation import MultiheadAttention

# Save a reference to the original forward method of MultiheadAttention
orig_forward = MultiheadAttention.forward

class Pretraining(nn.Module):
    def __init__(self, embedding_dim, nhead, dim_feedforward, num_layers,
                 dropout_prob, vocab_size, device, token_padding, seq_len):
        
        super().__init__()
        self.d_model = embedding_dim
        self.device = device
        self.vocab_size = vocab_size
        self.num_attention_layers = nhead
        self.seq_len = seq_len
        
        # Context embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx=token_padding).to(self.device)
        
        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(self.d_model, dropout_prob)

        # Transformer encoder
        self.encoder_layers = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_prob, batch_first=True, activation='gelu')
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers=num_layers)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.BatchNorm1d(self.seq_len + 2),
            nn.Linear(self.d_model * 4, self.vocab_size)
        )
        
        self.init_weights()
        
    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)
    
    def forward(self, X):
        context_embedding = self.embedding(X)
        pos_encoding = self.pos_encoder(context_embedding)
        encoder_output = self.transformer_encoder(pos_encoding)
        logits = self.mlp(encoder_output)
        logits = logits.view(-1, self.vocab_size)
        return logits
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 16000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
