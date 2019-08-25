import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import json
from sklearn.externals import joblib
from torch.nn.utils.rnn import pack_sequence
from torch.nn.parameter import Parameter


class DAT_GRU(nn.Module): 
    def __init__(self, args):
        super(DAT_GRU, self).__init__()
        self.model_name = 'DAT_GRU'
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
  
        self.lstm =nn.GRU(input_size = word_embed_dim,
                            hidden_size = hidden_size,
                            num_layers = 1,
                            bias = True,
                            batch_first = False,
                            bidirectional = False
                            )
        
        self.W_a = Parameter(rand_init(hidden_size, hidden_size+aspect_embed_dim))
        self.b_a = Parameter(rand_init(hidden_size, 1).squeeze())
        self.w = Parameter(rand_init(1, hidden_size))
        self.W_a2 = Parameter(rand_init(hidden_size, hidden_size+aspect_embed_dim))
        self.b_a2 = Parameter(rand_init(hidden_size, 1).squeeze())
        self.w2 = Parameter(rand_init(1, hidden_size))

        self.attn_softmax = nn.Softmax(dim=1)
        self.W_s = Parameter(rand_init(1, hidden_size+aspect_embed_dim))
        self.b_s = Parameter(rand_init(1, 1).squeeze())
        self.W_s2 = Parameter(rand_init(1, hidden_size+aspect_embed_dim))
        self.b_s2 = Parameter(rand_init(1, 1).squeeze())

        self.identity = torch.eye(128, 128)
        if args.cuda:
            self.identity = self.identity.cuda()

    def forward(self, content, aspect, mask):
        content = self.word_encoder(content)

        leng = content.size(0)
        
        aspect_embeddings = self.aspect_embed(aspect)
        aspect_embedding = aspect_embeddings.expand(-1, content.size()[0], -1)
        content_out, _ = self.lstm(content)

        aspect_embedding = aspect_embedding.permute(1,0,2)

        M = torch.tanh(F.linear(torch.cat((content_out, aspect_embedding), dim=2), self.W_a) + self.b_a)

        M = F.linear(M, self.w).squeeze(2)

        M = M.t()
        M = M + mask

        weights = self.attn_softmax(M).unsqueeze(1)

        r = torch.bmm(weights, content_out.permute(1, 0, 2)).squeeze(1)


        M2 = torch.tanh(F.linear(torch.cat((content_out, aspect_embedding), dim=2), self.W_a2) + self.b_a2)
        M2 = F.linear(M2, self.w2).squeeze(2)
        M2 = M2.t()
        M2 = M2 + mask
        weights2 = self.attn_softmax(M2).unsqueeze(1)
        r2 = torch.bmm(weights2, content_out.permute(1, 0, 2)).squeeze(1)

        orthogonal = torch.cat((weights, weights2), dim=1)

        orthogonal = torch.bmm(orthogonal.permute(0, 2, 1), orthogonal) 
        orthogonal = orthogonal - self.identity[:leng,:leng].expand(orthogonal.size(0), -1, -1)
        orthogonal_loss = torch.norm(orthogonal) / leng

        # kl1 = torch.log(weights / weights2) * weights
        # kl1 [kl1!=kl1] = 0
        # kl2 = torch.log(weights2 / weights) * weights2
        # kl2 [kl2!=kl2] = 0
        # kl = torch.sum(kl1) + torch.sum(kl2)
        # kl = kl 

        logits = F.linear(torch.cat((r, aspect_embeddings.squeeze(1)), dim=1), self.W_s, self.b_s)
        logits2 = F.linear(torch.cat((r2, aspect_embeddings.squeeze(1)), dim=1), self.W_s2, self.b_s2)
        logits = torch.cat((logits, logits2), dim=1)

        return logits, orthogonal_loss#.item()
    

def rand_init(row, col, grad=True):
    weight = torch.empty(row, col, dtype=torch.float32, requires_grad=grad)
    nn.init.normal_(weight, mean=0, std=0.01)
    return weight.cuda()
