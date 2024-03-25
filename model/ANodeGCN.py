import torch
import torch.nn.functional as F
import torch.nn as nn


class ANodeGCN(nn.Module):
    def __init__(self, dim_in, dim_out, embed_dim, cheb_k=2):
        super(ANodeGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  # N, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("nm,bmc->bnc", supports, x)  # B, N, dim_in
        x_gconv = torch.einsum('bni,nio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv

