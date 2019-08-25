import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        A = args.aspect_num

        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.embed.weight = nn.Parameter(args.embedding, requires_grad=True)
        self.aspect_embed = nn.Embedding(A, args.aspect_embed_dim)
        self.aspect_embed.weight = nn.Parameter(args.aspect_embedding, requires_grad=True)
        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K-2) for K in [3]])
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc_aspect = nn.Linear(100, Co)


    def forward(self, feature, aspect):
        feature = self.embed(feature)
        aspect_v = self.aspect_embed(aspect)
        aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)
        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit, x, y
