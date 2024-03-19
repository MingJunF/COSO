from configparser import ConfigParser
import torch
import torch.nn as nn
import numpy as np
import os
from torch.distributions import Normal
from cosoutils import AutoRegressiveLSTM, InfoNCE, CLUB, compute_sequence_length
from tqdm import tqdm
import torch.optim as optim

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
        self.trainable_init_input_confounder = nn.Parameter(torch.zeros(1, 1, self.num_covariates+self.num_treatments)).cuda(args.cuda_id)
        # Removed IV initialization
        self.trainable_h0_confounder, self.trainable_c0_confounder, self.trainable_z0_confounder = self.trainable_init_h_confounder()
        # Removed IV LSTM initialization

        # Independent mutual information which includes the loss function
        self.term_a = InfoNCE(self.num_covariates+self.num_confounders+self.num_S, self.num_treatments)
        #self.term_a = InfoNCE(self.num_covariates+self.num_confounders+self.num_S, self.num_treatments)
        self.term_b = InfoNCE(self.num_covariates+self.num_confounders+self.num_treatments, self.num_outcomes)
        self.term_S = CLUB(self.num_covariates+self.num_confounders+self.num_treatments+self.num_S, self.num_covariates+self.num_confounders+self.num_treatments+self.num_outcomes)
        # Removed IV related terms

        # Confounder LSTM generation
        self.lstm_confounder = AutoRegressiveLSTM(input_size=self.num_covariates + self.num_treatments,
                                                hidden_size=self.lstm_hidden_units,
                                                output_size=self.num_confounders,
                                                ).cuda(args.cuda_id)

        # Prediction for each treatment
        # 动态创建confounder解码器
        self.confounder_decoders = []
        for _ in range(self.num_treatments):
            confounder_decoder = nn.Sequential(
                nn.Linear(self.num_covariates + self.num_confounders, self.fc_hidden_units),
                nn.ELU(),
                nn.Linear(self.fc_hidden_units, self.fc_hidden_units // 2),
                nn.ELU(),
                nn.Linear(self.fc_hidden_units // 2, 1)
            ).cuda(args.cuda_id)
            self.confounder_decoders.append(confounder_decoder)
        self.to(torch.device(f'cuda:{args.cuda_id}'))



    def trainable_init_h_confounder(self):
        h0 = torch.zeros(1, self.lstm_hidden_units)
        c0 = torch.zeros(1, self.lstm_hidden_units)
        z0 = torch.zeros(1, self.num_confounders)
        trainable_h0 = nn.Parameter(h0, requires_grad=True)
        trainable_c0 = nn.Parameter(c0, requires_grad=True)
        trainable_z0 = nn.Parameter(z0, requires_grad=True)
        return trainable_h0, trainable_c0, trainable_z0


    def forward(self, previous_covariates, previous_treatments, current_covariates, batch_S):

        batch_size = previous_covariates.size(0)
        previous_covariates_and_treatments = torch.cat([previous_covariates, previous_treatments], -1).permute(1, 0, 2)
        lstm_input_confounder = torch.cat([self.trainable_init_input_confounder.repeat(1, batch_size, 1), previous_covariates_and_treatments], dim=0)
        lstm_input_confounder = lstm_input_confounder.float()
        sequence_lengths=compute_sequence_length(lstm_input_confounder)
        indexS=batch_S[0][0][0]
        # 假设 indexS 已经是一个整数或者numpy数组
        indexS_tensor = torch.tensor(indexS, dtype=torch.long)
        # 假设 lstm_input_confounder 是你想从中索引的Tensor
        S_feature = lstm_input_confounder[:, :, indexS_tensor]
        S = S_feature.view(-1, 1).float()

        # Removed IV input preparation
        # Generate confounder
        lstm_output_confounder, _ = self.lstm_confounder(inputs=lstm_input_confounder, 
                                                        initial_state=(self.trainable_h0_confounder.repeat(batch_size, 1),
                                                                        self.trainable_c0_confounder.repeat(batch_size, 1),
                                                                        self.trainable_z0_confounder.repeat(batch_size, 1)),
                                                        sequence_lengths=sequence_lengths)
        # Removed IV output generation

        # Definition of confounders
        hidden_confounders = lstm_output_confounder.view(-1, self.num_confounders)
        current_covariates = current_covariates.reshape(-1, self.num_covariates)

        multitask_input_confounder = torch.cat([hidden_confounders, current_covariates], dim=-1).float()
        confounder_pred_treatments = []
        for treatment in range(self.num_treatments):
            confounder_pred_treatments.append(self.confounder_decoders[treatment](multitask_input_confounder))
        confounder_pred_treatments = torch.cat(confounder_pred_treatments, dim=-1).float()


        return confounder_pred_treatments.view(-1, self.num_treatments), hidden_confounders.view(-1, self.num_confounders), S
        

    # Removed inference predict confounder and IV function
    def compute_hidden_confounders(self, dataset, args):
        confounders = []
        self.eval()  # 将模型设置为评估模式
        for (batch_previous_covariates, batch_previous_treatments, batch_current_covariates,
             batch_target_treatments, batch_outcomes,batch_S) in self.gen_epoch(dataset,args):
            # 使用模型的前向传播计算confounders
            _, confounder, S= self.forward(batch_previous_covariates, batch_previous_treatments, batch_current_covariates,batch_S)
            
            # 重新整形并转换数据，准备收集
            confounder = confounder.reshape(-1, self.num_confounders)
            confounder = confounder.cpu().detach().numpy()
            
            # 收集所有批次的confounders
            confounders.append(confounder)
        
        # 合并所有批次的confounders为一个numpy数组并返回
        return np.concatenate(confounders, axis=0)
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