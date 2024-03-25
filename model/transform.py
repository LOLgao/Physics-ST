import torch
import torch.nn as nn
import torch.nn.functional as F


class Transform(nn.Module):
    def __init__(self, time_channels, hidden_channels, num_head=8):
        super(Transform, self).__init__()

        # self.dropout = nn.Dropout(p=dropout_prob)

        self.num_head = num_head
        self.d = hidden_channels // num_head
        self.FC_q = nn.Linear(time_channels, hidden_channels)

        self.FC_k = nn.Linear(time_channels, hidden_channels)
        self.FC_v = nn.Linear(hidden_channels, hidden_channels)

        # self.FC = nn.Linear(num_for_target * hidden_channels, hidden_channels)

    def forward(self, encoder_hidden, x_time, target_time):
        '''

        :param encoder_hidden:  [batch_size, seq_time, node_num, hidden_channels]
        :param x_time: [batch, seq_time, time_channels]
        :param target_time: [batch, horizon, time_channels]
        :return:
        '''

        batch_size, node_num, seq_time, hidden_channels = encoder_hidden.shape

        query = self.FC_q(target_time)  # 未来嵌入特征用处查询   查找出历史的哪些与未来相似
        key = self.FC_k(x_time)  # 历史嵌入特征用于被查询，
        value = self.FC_v(encoder_hidden).transpose(1, 2)  # 历史流量用于加权，转换到未来

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)  # [K * batch_size, num_pred, d]
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0).transpose(1, 2)  # [K * batch_size, d, num_his]

        # [K * batch_size, num_nodes, num_for_predict, d]
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        # 时间注意力，因此  空间和时间维度需要调换
        attention = torch.matmul(query, key)  # [K * batch_size, num_pred, num_for_predict]
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)  # [K * batch_size, num_pred, num_for_predict]

        X = torch.einsum('nij,nmjd->nmid', attention, value)  # [K * batch_size, num_nodes, num_pred, d]

        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # [batch_size, num_nodes, num_pred, d * K]

        # [batch_size, num_nodes, num_pred, hidden] -> [batch_size, num_pred, num_nodes, hidden]
        X = X.transpose(1, 2)
        # X = self.FC(X.reshape(batch_size, node_num, -1))

        # X = F.relu(X)

        del query, key, value

        return X


