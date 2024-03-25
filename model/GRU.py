import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from lib.metrics import masked_mae_loss


# https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, num_nodes, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias).to('cuda')
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias).to('cuda')
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_size).to('cuda')

    def forward(self, x, hidden):

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        # gate_x = gate_x.squeeze()
        # gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)

        # 公式1
        resetgate = F.sigmoid(i_r + h_r)
        # 公式2
        inputgate = F.sigmoid(i_i + h_i)
        # 公式3
        newgate = F.tanh(i_n + (resetgate * h_n))
        # 公式4，不过稍微调整了一下公式形式
        hy = newgate + inputgate * (hidden - newgate)

        return hy


class RNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, num_layers):
        super(RNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GRUCell(node_num, dim_in, dim_out))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GRUCell(node_num, dim_out, dim_out))

    def forward(self, x, init_state):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :],
                                            state
                                            )
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.loss_func = args.loss_func
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.lag = args.lag
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.g_encoder = RNN(args.num_nodes, self.input_dim, args.rnn_units,
                             args.num_layers)

        self.end_conv = nn.Linear(self.hidden_dim, self.output_dim * self.horizon)

    def forward(self, source, condition, disease):
        hidden_init = self.g_encoder.init_hidden(source.shape[0])
        g_output = self.g_encoder(source, hidden_init)[:, -1, ...]
        # B, horizon, N, hidden
        output = self.end_conv(g_output)
        return output.transpose(1, 2).unsqueeze(3)

    def loss(self, outputs, label, device='cuda'):
        if self.loss_func == 'mask_mae':
            loss_pred = masked_mae_loss(mask_value=None)
        elif self.loss_func == 'mae':
            loss_pred = torch.nn.L1Loss().to(device)
        elif self.loss_func == 'mse':
            loss_pred = torch.nn.MSELoss().to(device)
        else:
            raise ValueError
        loss_1 = loss_pred(outputs, label)
        sep_loss = [loss_1.item()]
        return loss_1, sep_loss
