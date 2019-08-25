import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import json
from sklearn.externals import joblib
from torch.nn.utils.rnn import pack_sequence
from torch.nn.parameter import Parameter


class AT_LSTM(nn.Module): 
    def __init__(self, args):
        super(AT_LSTM, self).__init__()
        self.model_name = 'AT_LSTM'
        self.args = args
        
        V = args.embed_num
        word_embed_dim, hidden_size, aspect_embed_dim = args.embed_dim, args.embed_dim, args.embed_dim
        class_num = args.class_num
        A = args.aspect_num

        if len(args.aspect_embedding) < 2:
            self.aspect_embed = nn.Embedding.from_pretrained(rand_init(A, args.aspect_embed_dim), freeze=False)
        else:
            self.aspect_embed = nn.Embedding(A, args.aspect_embed_dim)
            self.aspect_embed.weight = nn.Parameter(args.aspect_embedding, requires_grad=True)
            
        self.word_encoder = nn.Embedding(V, word_embed_dim, padding_idx=1)
        self.word_encoder.weight = nn.Parameter(args.embedding, requires_grad=True)
        
        self.lstm =nn.LSTM(input_size = word_embed_dim,
                            hidden_size = hidden_size,
                            num_layers = 1,
                            bias = True,
                            batch_first = False,
                            bidirectional = False
                            )
        
        self.W_h = Parameter(rand_init(hidden_size, hidden_size))
        self.W_v = Parameter(rand_init(aspect_embed_dim, aspect_embed_dim))

        self.w = Parameter(rand_init(1, hidden_size*2))
        self.W_p = Parameter(rand_init(hidden_size, hidden_size))
        self.W_x = Parameter(rand_init(hidden_size, hidden_size))

        self.attn_softmax = nn.Softmax(dim=1)
        self.W_s = Parameter(rand_init(class_num, hidden_size))
        self.b_s = Parameter(rand_init(class_num, 1).squeeze())

    def forward(self, content, aspect, mask):
        content = self.word_encoder(content)
        content_out, (ht, ct) = self.lstm(content)
        aspect_embeddings = self.aspect_embed(aspect)
        aspect_embedding = aspect_embeddings.expand(-1, content_out.size()[0], -1)
        aspect_embedding = aspect_embedding.permute(1,0,2)
        M = torch.tanh(torch.cat((F.linear(content_out, self.W_h), F.linear(aspect_embedding, self.W_v)), dim=2))
        M = F.linear(M, self.w).squeeze(2)
        M = M.t()
        M = M + mask
        weights = self.attn_softmax(M)
        weights = weights.unsqueeze(1)
        r = torch.bmm(weights, content_out.permute(1, 0, 2)).squeeze(1)
        h = torch.tanh(torch.add(F.linear(r, self.W_p), F.linear(ht[0], self.W_x)))
        logits = F.linear(h, self.W_s, self.b_s)
        return logits


def rand_init(row, col, grad=True):
    weight = torch.empty(row, col, dtype=torch.float32, requires_grad=grad)
    nn.init.normal_(weight, mean=0, std=0.01)
    return weight.cuda()
