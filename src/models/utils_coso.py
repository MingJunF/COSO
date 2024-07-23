import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
def compute_sequence_length(sequence):
    used = torch.sign(torch.max(torch.abs(sequence), dim=2)[0])

    length = torch.sum(used, dim=1)
    length = length.long() 

    return length
class ConditionalFenchelMIUpper(nn.Module):
    def __init__(self, a_dim, b_dim, c_dim):
        super(ConditionalFenchelMIUpper, self).__init__()
        input_dim = a_dim + b_dim + 2 * c_dim
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, a, b, c, mask=None):
        ac = torch.cat((a, c), dim=1)
        bc = torch.cat((b, c), dim=1)
        joint = torch.cat((ac, bc), dim=1)

        marg_b = torch.roll(b, 1, dims=0)
        marg_bc = torch.cat((marg_b, c), dim=1)
        marg = torch.cat((ac, marg_bc), dim=1)

        if mask is not None:
            mask = mask.squeeze().bool()
            joint = joint[mask]
            marg = marg[mask]

        t_joint = self.fc1(joint)
        t_joint = F.relu(t_joint)
        t_joint = self.fc2(t_joint)

        t_marg = self.fc1(marg)
        t_marg = F.relu(t_marg)
        t_marg = self.fc2(t_marg)

        mi_upper_bound = torch.log(torch.mean(torch.exp(t_joint))) - torch.mean(t_marg)

        return torch.abs(mi_upper_bound)
class ConditionalMINE(nn.Module):
    def __init__(self, a_dim, b_dim, c_dim):
        super(ConditionalMINE, self).__init__()
        input_dim = a_dim + b_dim + 2 * c_dim
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, a, b, c, mask=None):
        ac = torch.cat((a, c), dim=1)
        bc = torch.cat((b, c), dim=1)
        joint = torch.cat((ac, bc), dim=1)

        marg_b = torch.roll(b, 1, dims=0)
        marg_bc = torch.cat((marg_b, c), dim=1)
        marg = torch.cat((ac, marg_bc), dim=1)

        if mask is not None:
            mask = mask.squeeze().bool()
            joint = joint[mask]
            marg = marg[mask]

        t = self.fc1(joint)
        t = F.relu(t)
        t = self.fc2(t)

        et = self.fc1(marg)
        et = F.relu(et)
        et = self.fc2(et)
        mi_lb = torch.mean(t) - torch.logsumexp(et, dim=0) / et.size(0)

        return torch.abs(mi_lb)
class MINE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_dim + output_dim, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x, y, mask=None):
        joint = torch.cat((x, y), dim=1)
        marg_y = torch.roll(y, 1, dims=0)
        marg = torch.cat((x, marg_y), dim=1)
        
        if mask is not None:
            mask = mask.squeeze().bool()
            joint = joint[mask]
            marg = marg[mask]

        t = self.fc1(joint)
        t = nn.ReLU()(t)
        t = self.fc2(t)

        et = self.fc1(marg)
        et = nn.ReLU()(et)
        et = self.fc2(et)

        # Use log-sum-exp trick for numerical stability
        log_mean_exp_et = torch.logsumexp(et, dim=0) - torch.log(torch.tensor(et.size(0), device=et.device, dtype=et.dtype))
        
        mi_lb = torch.mean(t) - log_mean_exp_et
        return mi_lb

class AutoRegressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_rate=0.0, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size+output_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

    def forward(self, x, sequence_length,initial_state=None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        outputs = []
        hn, cn, output_t = initial_state
        if initial_state is not None:
            hn, cn, output_t = initial_state
            hn = hn
            cn = cn
            output_t = output_t
        if self.training:
            h_dropout = torch.bernoulli(hn.data.new(hn.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
            c_dropout = torch.bernoulli(cn.data.new(cn.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
            out_dropout = torch.bernoulli(output_t.data.new(output_t.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
        for t in range(seq_len):
            effective_batch = (sequence_length > t).float().to(device)
            effective_batch = effective_batch.unsqueeze(-1)       
            lstm_input = torch.cat((x[:, t:t+1, :] * effective_batch, output_t), dim=-1)
            lstm_out, (hn, cn) = self.lstm(lstm_input, (hn, cn))
            if self.training:
                hn = hn * h_dropout
                cn = cn * c_dropout
            output_t = self.output_layer(lstm_out)
            if self.training:
                output_t = output_t*out_dropout
            outputs.append(output_t * effective_batch)
        outputs = torch.cat(outputs, dim=1)
        return outputs