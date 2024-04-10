from COSO_Factor_model import COSOFactorModel
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import tensorflow as tf
# device = 'cuda'

def train_model(factor_model, dataset_train, dataset_val, args):
    optimizer = optim.Adam(factor_model.parameters(), lr=factor_model.learning_rate)
    best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
    epochs_no_improve = 0  # 追踪没有改善的epoch数量
    for epoch in tqdm(range(factor_model.num_epochs)):
        factor_model.train()
        train_losses = []
        for (batch_previous_covariates, batch_previous_treatments, batch_current_covariates, batch_target_treatments, batch_outcomes, batch_S) in factor_model.gen_epoch(dataset_train, args):

            confounders, sequence_lengths,lstm_input_confounder= factor_model(batch_previous_covariates, batch_previous_treatments, batch_current_covariates)
            batch_size, time_steps, feature_size = batch_target_treatments.size()
            batch_current_covariates = batch_current_covariates.reshape(-1, factor_model.num_covariates).float()
            treatment_targets = batch_target_treatments.reshape(-1, factor_model.num_treatments).float()
            S =  batch_S.reshape(-1,1).float()
            outcomes = batch_outcomes.reshape(-1, 1).float()
            confounders = confounders.reshape(-1, factor_model.num_confounders).float()
            mask = torch.sign(torch.max(torch.abs(lstm_input_confounder), dim=2).values)
            flat_mask = mask.view(-1, 1)
            loss_a = factor_model.term_a(torch.cat([confounders,S], dim=-1), treatment_targets,mask=flat_mask)
            loss_b = factor_model.term_b(torch.cat([confounders, treatment_targets], dim=-1), outcomes,mask=flat_mask)
            loss_S = factor_model.term_S(torch.cat([confounders, treatment_targets,S], dim=-1), torch.cat([confounders, treatment_targets, outcomes], dim=-1),mask=flat_mask)
            #val_confounder_loss = compute_loss(factor_model,treatment_targets, confounder_pred_treatments, lstm_input_confounder, sequence_lengths)
            train_loss = loss_a+loss_b+args.alpha*loss_S
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())


        factor_train_loss = np.mean(train_losses)

        factor_model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_previous_covariates, batch_previous_treatments, batch_current_covariates,
                 batch_target_treatments, batch_outcomes,batch_S) in factor_model.gen_epoch(dataset_val,args):
                confounders, sequence_lengths,lstm_input_confounder= factor_model(batch_previous_covariates, batch_previous_treatments, batch_current_covariates)
                batch_current_covariates = batch_current_covariates.reshape(-1, factor_model.num_covariates).float()
                batch_size, time_steps, feature_size = batch_target_treatments.size()
                treatment_targets = batch_target_treatments.reshape(-1, factor_model.num_treatments).float()
                S =  batch_S.reshape(-1, 1).float()
                outcomes = batch_outcomes.reshape(-1, 1).float()
                confounders = confounders.reshape(-1, factor_model.num_confounders).float()
                mask = torch.sign(torch.max(torch.abs(lstm_input_confounder), dim=2).values)
                flat_mask = mask.view(-1, 1)
                loss_a = factor_model.term_a(torch.cat([confounders,S], dim=-1), treatment_targets,mask=flat_mask)
                loss_b = factor_model.term_b(torch.cat([confounders, treatment_targets], dim=-1), outcomes,mask=flat_mask)
                loss_S = factor_model.term_S(torch.cat([confounders, treatment_targets,S], dim=-1), torch.cat([confounders, treatment_targets, outcomes], dim=-1),mask=flat_mask)
                #val_confounder_loss = compute_loss(factor_model,treatment_targets, confounder_pred_treatments, lstm_input_confounder, sequence_lengths)
                val_loss = loss_a+loss_b+loss_S
                val_losses.append(val_loss)

            factor_val_loss = np.mean([loss.item() for loss in val_losses])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0  # 重置计数器
            else:
                epochs_no_improve += 1  # 没有改善则计数器加1

            # 如果连续5个epoch没有改善，则早停
            if epochs_no_improve == 5:
                print('Validation loss has not improved for 5 consecutive epochs. Stopping training.')
                break

        print(f"Epoch {epoch+1} ---- Train Loss: {factor_train_loss:.5f}, Val Loss: {factor_val_loss:.5f}")


def compute_loss(factor_model,target_treatments, treatment_predictions, lstm_input_confounder, sequence_length):
    # 将目标处理结果重塑以匹配预测的形状
    target_treatments_reshape = target_treatments.view(-1, factor_model.num_treatments)

    
        # 根据 rnn_input 生成掩码，以便在计算损失时忽略填充位置
    mask = torch.sign(torch.max(torch.abs(lstm_input_confounder), dim=2).values)
    flat_mask = mask.view(-1, 1)
    
    # 手动计算交叉熵损失，并进行裁剪以避免 log(0)
    clipped_predictions = torch.clamp(treatment_predictions, 1e-10, 1.0)
    clipped_inverse_predictions = torch.clamp(1.0 - treatment_predictions, 1e-10, 1.0)
    cross_entropy = -torch.sum(
        (target_treatments_reshape * torch.log(clipped_predictions) + 
         (1 - target_treatments_reshape) * torch.log(clipped_inverse_predictions)) * flat_mask, 
        dim=0)
    
    # 通过序列长度归一化
    norm = torch.sum(torch.tensor(sequence_length).float(), dim=0)
    cross_entropy /= norm
    
    # 在批次上取平均
    return torch.mean(cross_entropy)
