import torch
import torch.nn.functional as F
import torch.nn as nn


class ANodeRnnCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, embed_dim):
        super(ANodeRnnCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = ANodeGCN(dim_in + self.hidden_dim, 2 * dim_out, embed_dim)
        self.update = ANodeGCN(dim_in + self.hidden_dim, dim_out, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


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

