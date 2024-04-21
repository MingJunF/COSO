import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
def compute_sequence_length(sequence):

    # 计算每个时间步的最大绝对值，然后判断是否大于0
    used = torch.sign(torch.max(torch.abs(sequence), dim=2)[0])
    # 计算每个序列的实际长度
    length = torch.sum(used, dim=1)
    length = length.long()  # 确保长度为整数类型

    return length

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size=64):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples, mask):
        mask = mask.bool()

        # Apply mask
        x_samples = x_samples[mask.squeeze(), :]
        y_samples = y_samples[mask.squeeze(), :]
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # Improve numerical stability
        epsilon = 1e-8
        logvar = torch.clamp(logvar, min=-20, max=2)  # Limit logvar range to prevent overflow/underflow
        var = torch.exp(logvar) + epsilon  # Add a small constant before taking the exp to prevent underflow
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 / (2.0 * var)
        
        prediction_1 = mu.unsqueeze(1)  # shape [nsample, 1, dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1, nsample, dim]

        # log of conditional probability of negative sample pairs
        negative = ((y_samples_1 - prediction_1)**2).mean(dim=1) / (2.0 * var)

        return -(positive.sum(dim = -1) - negative.sum(dim = -1)).mean()


    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples,mask):
        mask = mask.bool()

        # Apply mask
        x_samples = x_samples[mask.squeeze(), :]
        y_samples = y_samples[mask.squeeze(), :]
        return - self.loglikeli(x_samples, y_samples)



class InfoNCE2(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=64):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())
    
    def forward(self, x_samples, y_samples,mask):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size))

        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=64):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())
    
    def forward(self, x_samples, y_samples, mask):  # mask has shape [batch_size*timestep, 1]
        # Ensure mask is a boolean tensor
        mask = mask.bool()

        # Apply mask
        x_samples_masked = x_samples[mask.squeeze(), :]
        y_samples_masked = y_samples[mask.squeeze(), :]
        sample_size = y_samples_masked.shape[0]

        # Since x_samples and y_samples are already [batch*timestep, feature], we adjust the tile operations accordingly
        x_tile = x_samples_masked.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples_masked.unsqueeze(1).repeat((1, sample_size, 1))
        # Concatenate and calculate T0 and T1 using masked samples
        T0 = self.F_func(torch.cat([x_samples_masked, y_samples_masked], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [sample_size, sample_size, 1]
        # Calculate lower bound
        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))

        return -lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class AutoRegressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_rate=0.4, batch_first=True):
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
        # 初始的输出（可以设为全零向量）
        # 自回归地处理每个时间步
        if self.training:
            h_dropout = torch.bernoulli(hn.data.new(hn.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
            c_dropout = torch.bernoulli(cn.data.new(cn.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
            out_dropout = torch.bernoulli(output_t.data.new(output_t.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
        for t in range(seq_len):
            # 只对序列的非填充部分生成输出
            effective_batch = (sequence_length > t).float().to(device)
            effective_batch = effective_batch.unsqueeze(-1)       
            lstm_input = torch.cat((x[:, t:t+1, :] * effective_batch, output_t), dim=-1)

            # LSTM前向传播
            lstm_out, (hn, cn) = self.lstm(lstm_input, (hn, cn))

            if self.training:
                hn = hn * h_dropout
                cn = cn * c_dropout

            # 计算当前时间步的输出
            output_t = self.output_layer(lstm_out)

            # 应用dropout
            if self.training:
                output_t = output_t*out_dropout

            # 更新输出
            outputs.append(output_t * effective_batch)

        # 将所有时间步的输出拼接起来
        outputs = torch.cat(outputs, dim=1)
        return outputs