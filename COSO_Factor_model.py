from configparser import ConfigParser
import torch
import torch.nn as nn
import numpy as np
import os
from torch.distributions import Normal
from cosoutils import AutoRegressiveLSTM, InfoNCE, CLUB, compute_sequence_length,VariationalLSTM
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
class COSOFactorModel(nn.Module):
    def __init__(self, params, hyperparams, args):
        super(COSOFactorModel, self).__init__()
        self.num_treatments = params['num_treatments']
        self.num_covariates = params['num_covariates']
        #self.num_raw_covariates = params['num_raw_covariates']
        self.num_confounders = params['num_confounders']
        self.num_outcomes = params['num_outcomes']
        self.max_sequence_length = params['max_sequence_length']
        self.num_epochs = params['num_epochs']
        self.num_S=params['num_S']
        self.lstm_hidden_units = hyperparams['lstm_hidden_units']
        self.fc_hidden_units = hyperparams['fc_hidden_units']
        self.learning_rate = hyperparams['learning_rate']
        self.batch_size = hyperparams['batch_size']
        self.args = args        
        self.trainable_init_input_confounder = nn.Parameter(torch.zeros(self.batch_size, 1, self.num_covariates)).cuda(args.cuda_id)
        # Removed IV initialization
        self.trainable_h0_confounder, self.trainable_c0_confounder, self.trainable_z0_confounder = self.trainable_init_h_confounder()
        # Removed IV LSTM initialization
        # Independent mutual information which includes the loss function
        self.term_a = InfoNCE(self.num_confounders+self.num_S, self.num_treatments)
        self.term_b = InfoNCE(self.num_confounders+self.num_treatments, self.num_outcomes)
        self.term_S = CLUB(self.num_confounders+self.num_treatments+self.num_S, self.num_confounders+self.num_treatments+self.num_outcomes)
        # Confounder LSTM generation
        self.lstm_confounder = VariationalLSTM(input_size=self.num_covariates,
                                                hidden_size=self.lstm_hidden_units,
                                                output_size=self.num_confounders
                                                ).cuda(args.cuda_id)

        # Prediction for each treatment
        # 动态创建confounder解码器
        self.confounder_decoders = nn.ModuleList()  # 使用ModuleList来存储所有解码器
        for _ in range(self.num_treatments):
            # 定义单个解码器网络
            confounder_decoder = nn.Sequential(
                nn.Linear(self.num_covariates+self.num_confounders, self.fc_hidden_units),
                nn.LeakyReLU(),
                nn.Linear(self.fc_hidden_units, 1),
                nn.Sigmoid()
            ).to(torch.device(f'cuda:{args.cuda_id}'))  # 确保模型在正确的设备上
            self.confounder_decoders.append(confounder_decoder)

        self.to(torch.device(f'cuda:{args.cuda_id}'))



    def trainable_init_h_confounder(self):
        h0 = torch.zeros(1, self.batch_size,self.lstm_hidden_units)
        c0 = torch.zeros(1,self.batch_size, self.lstm_hidden_units)
        z0 = torch.zeros(self.batch_size,1 ,self.num_confounders)
        trainable_h0 = nn.Parameter(h0, requires_grad=True)
        trainable_c0 = nn.Parameter(c0, requires_grad=True)
        trainable_z0 = nn.Parameter(z0, requires_grad=True)
        return trainable_h0, trainable_c0, trainable_z0


    def forward(self, previous_covariates, previous_treatments, current_covariates):
        batch_size = previous_covariates.size(0)
        previous_covariates_and_treatments = torch.cat([previous_covariates, previous_treatments], -1)
        init_input_confounder_sub = self.trainable_init_input_confounder[:batch_size, :, :]    
        lstm_input_confounder = torch.cat([current_covariates], dim=1)
        lstm_input_confounder = lstm_input_confounder.float()
        sequence_lengths=compute_sequence_length(lstm_input_confounder)
        hn = self.trainable_h0_confounder[:, :batch_size, :].contiguous()
        cn = self.trainable_c0_confounder[:, :batch_size, :].contiguous()
        zn = self.trainable_z0_confounder[:batch_size, :, :].contiguous()
        lstm_output_confounder = self.lstm_confounder(lstm_input_confounder, sequence_length=sequence_lengths, initial_state=(hn, cn, zn))


        # Definition of confounders
        hidden_confounders = lstm_output_confounder.view(-1, self.num_confounders)
        #current_covariates = current_covariates.reshape(-1, self.num_covariates).float()

        #multitask_input_confounder = torch.cat([hidden_confounders, current_covariates], dim=-1).float()
        #confounder_pred_treatments = []
        #for treatment in range(self.num_treatments):
            #confounder_pred_treatments.append(self.confounder_decoders[treatment](multitask_input_confounder))
        #confounder_pred_treatments = torch.cat(confounder_pred_treatments, dim=-1).float()

        return hidden_confounders,sequence_lengths,lstm_input_confounder
        

    # Removed inference predict confounder and IV function
    def compute_hidden_confounders(self, dataset, args):
        dataset_size = dataset['covariates'].shape[0]
        hidden_confounders = np.zeros((dataset_size, self.max_sequence_length, self.num_confounders))
        
        self.eval()  # 设置模型为评估模式
        total_processed = 0  # 已处理的数据总数
        
        num_samples = 50 # 蒙特卡罗采样次数

        for (batch_previous_covariates, batch_previous_treatments, batch_current_covariates, batch_target_treatments, batch_outcomes, batch_S) in self.gen_epoch(dataset, args):
            batch_size = batch_previous_covariates.size(0)
            batch_confounders = np.zeros((batch_size, self.max_sequence_length, self.num_confounders)) 
            for _ in range(num_samples):
                confounder, _, _ = self.forward(batch_previous_covariates, batch_previous_treatments, batch_current_covariates)
            # 打开文件以写入模式

                confounder = confounder.cpu().detach().numpy()
                confounder_reshaped = confounder.reshape(batch_size, self.max_sequence_length, self.num_confounders)
                batch_confounders += confounder_reshaped
            
            # 计算当前批次的平均混杂因子
            batch_confounders /= num_samples
            # 确保只更新当前批次对应的hidden_confounders部分
            hidden_confounders[total_processed:total_processed + batch_size, :, :] = batch_confounders
            
            total_processed += batch_size  # 更新已处理的数据总数

        return hidden_confounders

    def gen_epoch(self, dataset,args):
        dataset_size = dataset['previous_covariates'].shape[0]
        num_batches = int(np.ceil(dataset_size / self.batch_size))

        for i in range(num_batches):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, dataset_size)  # 确保end_index不会超出数据集大小

            device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')

            batch_previous_covariates = torch.from_numpy(dataset['previous_covariates'][start_index:end_index, :, :]).to(device)
            batch_previous_treatments = torch.from_numpy(dataset['previous_treatments'][start_index:end_index, :, :]).to(device)
            batch_current_covariates = torch.from_numpy(dataset['covariates'][start_index:end_index, :, :]).to(device)
            batch_target_treatments = torch.from_numpy(dataset['treatments'][start_index:end_index, :, :].astype(np.float32)).to(device)
            batch_outcomes = torch.from_numpy(dataset['outcomes'][start_index:end_index, :, :].astype(np.float32)).to(device)
            batch_S = torch.from_numpy(dataset['S'][start_index:end_index, :, :].astype(np.float32)).to(device)

            
            yield (batch_previous_covariates, batch_previous_treatments, batch_current_covariates,
                batch_target_treatments, batch_outcomes,batch_S)
            
    def create_sequence_mask(sequence_lengths, max_len=None, batch_size=None):
        if max_len is None:
            max_len = sequence_lengths.max()
        if batch_size is None:
            batch_size = sequence_lengths.size(0)
        mask = torch.arange(max_len, device=sequence_lengths.device).expand(batch_size, max_len) < sequence_lengths.unsqueeze(1)
        return mask
