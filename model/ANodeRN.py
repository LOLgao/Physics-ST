import math
import torch
import torch.nn as nn
from model.ANodeGCN import ANodeGCN
from model.GRU import RNN
from lib.metrics import masked_mae_loss
from model.transform import Transform
import torch.nn.functional as F

class ANodeDecoder(nn.Module):
    def __init__(self, gen_layers, device, batch_size, num_nodes, ada_dim, latent_dim, out_dim, c_t):
        super(ANodeDecoder, self).__init__()
        self.gen_layers = gen_layers
        self.device = device
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.ada_dim = ada_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.c_t = c_t
        self.embeddings_normal = nn.Parameter(torch.randn(self.num_nodes, self.ada_dim))
        # self.embeddings_disease = nn.Parameter(torch.randn(self.num_nodes, self.ada_dim))
        self.first_layers_q = nn.Sequential(nn.Linear(self.c_t, self.latent_dim))
        self.first_layers_qr = nn.Sequential(nn.Linear(self.c_t // 4, self.latent_dim))

        self.second_layers_q = nn.Linear(self.latent_dim, self.latent_dim)
        self.second_layers_qr = nn.Linear(self.latent_dim, self.latent_dim)

        self.gcn_layers_no = ANodeGCN(self.latent_dim, self.latent_dim, self.ada_dim).to(self.device)

        self.project = ANodeGCN(self.latent_dim, self.latent_dim, self.ada_dim).to(self.device)

    def forward(self, states):
        c, c_condition, c_disease, state = states

        Qt = torch.sigmoid(self.first_layers_q(c_condition.unsqueeze(1)))
        Qt = torch.tanh(self.second_layers_q(Qt * c))

        qr = torch.sigmoid(self.first_layers_qr(c_disease[:, :1].unsqueeze(1)))
        for n in range(1, c_disease.shape[1]):
            qr = qr + torch.sigmoid(self.first_layers_qr(c_disease[:, n:n + 1].unsqueeze(1)))
        qr = Qt + torch.tanh(self.second_layers_qr(qr * c))

        # theta = torch.sigmoid(self.project(c, self.embeddings_normal))
        theta = torch.cat(torch.split(state, self.out_dim, dim=-1), dim=0)
        c_t = torch.tanh(self.gcn_layers_no(c, self.embeddings_normal))
        c_t = torch.cat(torch.split(c_t, self.out_dim, dim=-1), dim=0)
        qr = torch.cat(torch.split(qr, self.out_dim, dim=-1), dim=0)

        attention = torch.matmul(theta, theta.transpose(1, 2))
        attention /= (self.out_dim ** 0.5)
        attention = F.softmax(attention, dim=-1)

        x = - torch.einsum('nij,njd->nid', attention, c_t)
        x = torch.cat(torch.split(x, self.batch_size, dim=0), dim=-1)  # [batch_size, num_nodes, num_pred, d * K]

        return x


class PhysicsST(nn.Module):
    def __init__(self, args):
        super(PhysicsST, self).__init__()
        self.no_transform = args.no_transform
        self.loss_func = args.loss_func
        self.num_node = args.num_nodes
        self.weights = args.weights
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.lag = args.lag
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.ms = args.ms
        self.default_graph = args.default_graph
        self.g_encoder = RNN(args.num_nodes, self.input_dim, args.rnn_units, args.num_layers)
        self.transform = Transform(args.c_t, args.rnn_units)
        self.mlp = nn.Linear(self.lag, self.horizon)
        # predictor
        self.decoder = ANodeDecoder(
            args.gen_layers,
            args.device,
            args.batch_size,
            args.num_nodes,
            args.embed_dim,
            args.rnn_units,
            args.rnn_units // self.ms,
            args.c_t
        )
        self.end_conv = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                      nn.Linear(self.hidden_dim, self.hidden_dim),
                                      nn.Linear(self.hidden_dim, self.output_dim))
        self.thm = TemporalHeteroModel(args.rnn_units, args.batch_size, args.num_nodes)

    def forward(self, source, condition, disease):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        hidden_init = self.g_encoder.init_hidden(source.shape[0])
        g_output = self.g_encoder(source, hidden_init)
        state = self.transform(g_output, condition[:, :self.lag, :], condition[:, -self.horizon:, :])
        g_output = self.mlp(g_output.transpose(1, 3)).transpose(1, 3)
        outputs = []
        results = []
        # B, horizon, N, hidden
        for t in range(self.horizon):
            output = self.decoder((g_output[:, t, ...],
                                   condition[:, self.lag + t, :],
                                   disease[:, self.lag + t, :], state[:, t, ...]))
            output = g_output[:, t, ...] + output
            results.append(self.end_conv(output))
            outputs.append(output)
        outputs = torch.concat(outputs, dim=2)
        results = torch.concat(results, dim=2)

        return outputs.transpose(1, 2).unsqueeze(3), results.transpose(1, 2).unsqueeze(3)

    def loss(self, outputs, results, label, device='cuda'):
        if self.loss_func == 'mask_mae':
            loss_pred = masked_mae_loss(mask_value=None)
        elif self.loss_func == 'mae':
            loss_pred = torch.nn.L1Loss().to(device)
        elif self.loss_func == 'mse':
            loss_pred = torch.nn.MSELoss().to(device)
        else:
            raise ValueError
        loss = loss_pred(results, label)
        sep_loss = [loss.item()]

        l2 = self.temporal_loss(outputs)
        sep_loss.append(l2.item())
        loss += self.weights * l2

        return loss, sep_loss

    def temporal_loss(self, z):
        return self.thm(z)


class TemporalHeteroModel(nn.Module):
    '''Temporal heterogeneity modeling in a contrastive manner.
    '''

    def __init__(self, c_in, batch_size, num_nodes, device='cuda'):
        super(TemporalHeteroModel, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(num_nodes, c_in))  # representation weights
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()

        lbl_rl = torch.ones(batch_size, num_nodes)
        lbl_fk = torch.zeros(batch_size, num_nodes)
        lbl = torch.cat((lbl_rl, lbl_fk), dim=1)
        if device == 'cuda':
            self.lbl = lbl.cuda()

        self.n = batch_size

    def forward(self, z):
        '''
        :param z (tensor): shape nlvc, i.e., (batch_size, seq_len, num_nodes, feat_dim)
        :return loss: loss of generative branch. nclv
        '''
        z = torch.mean(z, dim=1)
        h = (z * self.W)  # nlvc
        s = torch.mean(h, dim=1)  # nlc
        s = self.sigm(s)  # s: summary of h, nc
        # select another region in batch
        idx = torch.randperm(self.n)
        shuf_h = h[idx]

        logits = self.disc(s, h, shuf_h)
        loss = self.b_xent(logits, self.lbl)
        return loss


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.net = nn.Bilinear(n_h, n_h, 1)  # similar to score of CPC

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, summary, h_rl, h_fk):
        '''
        :param s: summary, (batch_size, seq_len, feat_dim)
        :param h_rl: real hidden representation (w.r.t summary),
            (batch_size, seq_len, num_nodes, feat_dim)
        :param h_fk: fake hidden representation
        :return logits: prediction scores, (batch_size, seq_len, num_nodes, 2)
        '''
        s = torch.unsqueeze(summary, dim=1)
        s = s.expand_as(h_rl).contiguous()

        # score of real and fake, (batch_size, seq_len, num_nodes)
        sc_rl = torch.squeeze(self.net(h_rl, s), dim=2)
        sc_fk = torch.squeeze(self.net(h_fk, s), dim=2)

        logits = torch.cat((sc_rl, sc_fk), dim=1)

        return logits
