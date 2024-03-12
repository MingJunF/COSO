import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AutoRegressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(AutoRegressiveLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = nn.LSTMCell(self.input_size + self.output_size, self.hidden_size)
        self.fc = nn.Sequential(nn.Linear(in_features=self.hidden_size,out_features=self.output_size), nn.Tanh())
        self.dropout_rate = dropout_rate

    def forward(self, inputs, initial_state=None, sequence_lengths=None):
        if initial_state is None:
            raise ValueError("Initial state must be provided.")

        time_steps, batch_size, _ = inputs.size()
        outputs = []
        h, c, z = initial_state

        # 生成dropout掩码，用于隐藏状态h和细胞状态c
        dropout_mask_h = torch.bernoulli(torch.full((batch_size, self.hidden_size), 1 - self.dropout_rate)).to(inputs.device) / (1 - self.dropout_rate)
        dropout_mask_c = torch.bernoulli(torch.full((batch_size, self.hidden_size), 1 - self.dropout_rate)).to(inputs.device) / (1 - self.dropout_rate)

        for t in range(time_steps):
            combined_input = torch.cat([inputs[t], z], dim=1)
            h, c = self.cell(combined_input, (h, c))
            
            # 应用dropout掩码到隐藏状态h和细胞状态c
            if self.training:
                h = h * dropout_mask_h
                c = c * dropout_mask_c

            # Update z only if the sequence is not completed
            active = (t < sequence_lengths).float().unsqueeze(1) if sequence_lengths is not None else 1
            z = self.fc(h) * active

            # 应用dropout到输出z
            if self.training:
                out_dropout = torch.bernoulli(torch.full_like(z, 1 - self.dropout_rate)).to(inputs.device) / (1 - self.dropout_rate)
                z *= out_dropout

            outputs.append(z.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # Concatenate along the time dimension
        return outputs, (h, c)








def compute_sequence_length(sequence):
    """
    计算给定序列的实际长度。
    参数:
        sequence: Tensor, 形状为 [time_steps, batch_size, input_size]
    返回:
        length: Tensor, 形状为 [batch_size], 包含每个序列的实际长度
    """
    # 计算每个时间步的最大绝对值，然后判断是否大于0
    used = torch.sign(torch.max(torch.abs(sequence), dim=2)[0])
    # 计算每个序列的实际长度
    length = torch.sum(used, dim=0)
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
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
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
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)



class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=64):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
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