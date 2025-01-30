import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class FineTuningNN(nn.Module):
    def __init__(self, embedding_dim, nhead, dim_feedforward, num_layers,
                 dropout_prob_1, dropout_prob_2, vocab_size, device,
                 token_padding, unique_apis, num_classes, num_sequences,
                 cnn_kernel_stride, cnn_kernel_size, maxpool_kernel_stride,
                 maxpool_kernel_size, cnn_output_channels):
        
        super().__init__()
        self.d_model = embedding_dim
        self.device = device
        self.vocab_size = vocab_size
        self.unique_apis = unique_apis
        self.num_classes = num_classes
        self.num_sequences = num_sequences
        self.padding_token = token_padding

        # Context embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx=self.padding_token).to(self.device)
        
        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(self.d_model, dropout_prob_1)

        # Transformer encoder
        self.encoder_layers = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_prob_1, batch_first=True, activation='gelu')
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers=num_layers)

        # Sequence attention
        self.sequence_attention = nn.Linear(self.d_model, self.d_model)
        self.u_a = nn.Linear(self.d_model, 1)
        self.softmax = nn.Softmax(dim=1)

        # CNN parameters
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=cnn_output_channels, kernel_size=(8, self.d_model), stride=1, padding=0),
            nn.BatchNorm2d(cnn_output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=maxpool_kernel_stride),
            nn.Flatten()
        )
        
        self.fully_connected_inp = self.calculate_dim_of_cnn()
        self.dense = nn.Linear(self.fully_connected_inp, 64)
        self.classification = nn.Linear(64, 1)
        self.dropout_layer = nn.Dropout(dropout_prob_2)
        self.init_weights()

    def calculate_dim_of_cnn(self):
        inp_cnn = torch.randn(1, 1, self.num_sequences, self.d_model)
        cnn_out = self.conv2(inp_cnn)
        return cnn_out.shape[1]

    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)
    
    def forward(self, input_data):
        token_embedding = self.embedding(input_data)
        N_samples, Batch, sequence_length, embed = token_embedding.shape
        token_embedding_reshaped = token_embedding.view(N_samples * Batch, sequence_length, embed)
        positional_embedding = self.pos_encoder(token_embedding_reshaped)
        
        encoder_output = self.transformer_encoder(positional_embedding)
        encoder_output = encoder_output.view(N_samples, Batch, sequence_length, self.d_model)
        mean_across_api = torch.mean(encoder_output, dim=2)
        
        sequence_vector = torch.tanh(self.sequence_attention(mean_across_api))
        sequence_attention_sum = self.u_a(sequence_vector).squeeze(2)
        sequence_attention = torch.softmax(sequence_attention_sum, dim=1)
        
        sequence_sum_vector = sequence_attention.unsqueeze(2) * mean_across_api
        unsqueezed_inp = sequence_sum_vector.unsqueeze(1)
        
        cnn_out = self.conv2(unsqueezed_inp)
        out = self.dense(cnn_out)
        logits = self.classification(out).squeeze(1)
        return logits, sequence_attention

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
