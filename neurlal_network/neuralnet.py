import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from math import floor
from torch.nn.modules.activation import MultiheadAttention

# Save a reference to the original forward method of MultiheadAttention
orig_forward = MultiheadAttention.forward

class Pretraining(nn.Module):
    def __init__(self, embedding_dim, nhead, dim_feedforward, num_layers,
                dropout_prob, vocab_size, device, token_padding, unique_apis):

        super().__init__()
        # model parameters
        self.d_model = embedding_dim
        self.num_encoder_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob
        self.device = device
        self.vocab_size = vocab_size
        self.unique_apis = unique_apis 
        self.num_attention_layers = nhead
        #context embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx=token_padding).to(self.device)
        #positional encoding layer
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout_prob)
        #initializing transformer encoder layer
        self.encoder_layers = TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_attention_layers, dim_feedforward=self.dim_feedforward, dropout=self.dropout_prob, batch_first=True, activation='gelu')
        #initializing number of transformer encoder layers
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers=self.num_encoder_layers)
        self.decoder = nn.Linear(self.d_model, self.unique_apis)
        
        #self.dropout = nn.Dropout(self.dropout_prob)
        self.init_weights()
        
    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform(self.weight)
            self.bias.data.fill_(0.01)


    def forward(self, X):
        #shape of X: [batch_size, sequence_budget, api_budget]
        #context_embedding: [batch_size*sequence_budget, api_padding, embedding_dimention]
        context_embedding = self.embedding(X)
        #pos_embedding: [batch_size*sequence_budget, api_padding, embedding_dimention]
        pos_encoding = self.pos_encoder(context_embedding)
        #encoder_out :[batch_size * sequence_budget, api_padding, embedding_dimention]
        encoder_output = self.transformer_encoder(pos_encoding)
        flattened_encoder_output = encoder_output.view(-1, encoder_output.shape[-1])
        logits = self.decoder(flattened_encoder_output)
        return logits
        
        


class FineTuining(nn.Module):
    def __init__(self, embedding_dim, nhead, dim_feedforward, num_layers,
                 dropout_prob_1,  dropout_prob_2, vocab_size, device, 
                 token_padding, unique_apis, num_classes, num_sequences,
                 cnn_kernel_stride,cnn_kernel_size, maxpool_kernel_stride, 
                 maxpool_kernel_size, cnn_output_channels):

        super().__init__()
        #self.sequence_size = sequence_padding
        self.d_model = embedding_dim
        self.num_encoder_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_prob_1 = dropout_prob_1
        self.dropout_prob_2 = dropout_prob_2
        self.device = device
        self.vocab_size = vocab_size
        self.unique_apis = unique_apis 
        self.num_attention_layers = nhead
        self.num_classes = num_classes
        self.num_sequences = num_sequences
        self.padding_token = token_padding
        #context embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx= self.padding_token).to(self.device)
        #positional encoding layer
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout_prob_1)
        #initializing transformer encoder layer
        self.encoder_layers = TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_attention_layers, dim_feedforward=self.dim_feedforward, dropout=self.dropout_prob_1, batch_first=True, activation='gelu')
        #initializing number of transformer encoder layers
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers=self.num_encoder_layers)
        #New Attention Approach
        self.sequence_attention = nn.Linear(self.d_model, self.d_model)
        self.u_a = nn.Linear(self.d_model, 1)
        self.softmax = nn.Softmax(dim=1)

        #CNN parameters
        self.cnn_stride = cnn_kernel_stride
        self.cnn_kernel_size = cnn_kernel_size
        
        self.maxpool_stride = maxpool_kernel_stride
        self.maxpool_kernel_size = maxpool_kernel_size
        self.inp_channels = self.num_sequences
        self.out_channels = cnn_output_channels
        self.cnn_padding = int((self.cnn_kernel_size - 1) / 2)
        
        # CNN 1D Layer
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(8,768), stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=self.maxpool_stride),
            nn.Flatten(),)
        self.fully_connected_inp = self.calculate_dim_of_cnn()
        self.dense = nn.Linear( self.fully_connected_inp, 64)
        self.classification = nn.Linear(64, 1)
        
        # # self.classification_layer = nn.Linear(self.fully_connected_out, 1)
        self.dropout_layer = nn.Dropout(self.dropout_prob_2)
        
        self.init_weights()

    def calculate_dim_of_cnn(self):
        inp_cnn = torch.randn(1, 1, 350, 768)
        cnn_out = self.conv2(inp_cnn)
        return cnn_out.shape[1]

    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform(self.weight)
            self.bias.data.fill_(0.01)
        

    def forward(self, input_data):
        # Token Embedding
        token_embedding = self.embedding(input_data)
        positional_embedding = self.pos_encoder(token_embedding)
        # Transformer encoder
        encoder_output = self.transformer_encoder(positional_embedding)
        # Taking mean across API dimension
        mean_across_api = torch.mean(encoder_output, dim=1)
        sequence_vector = torch.tanh(self.sequence_attention(mean_across_api))
        sequence_attention_sum = self.u_a(sequence_vector).squeeze(1)
        sequence_attention = torch.softmax(sequence_attention_sum, dim=0)
        #sequence_attention dim: S
        sequence_sum_vector = sequence_attention.unsqueeze(1) * mean_across_api
        #Ssequence_sum_vector dim: SxE
        #With CNN
        unsqueezed_inp = sequence_sum_vector.unsqueeze(0)
        unsqueezed_inp = unsqueezed_inp.unsqueeze(0)
        
        cnn_out =  self.conv2(unsqueezed_inp)
        out = cnn_out.flatten()
        out = self.dense(out)
        logits = self.classification(out)
        
        return logits, sequence_attention

    #taken from pytoch implementation
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
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

    



        

    