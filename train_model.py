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

    for epoch in tqdm(range(factor_model.num_epochs)):
        factor_model.train()
        train_losses = []
        for (batch_previous_covariates, batch_previous_treatments, batch_current_covariates, batch_target_treatments, batch_outcomes, batch_S) in factor_model.gen_epoch(dataset_train, args):
            confounder_pred_treatments, confounders, S= factor_model(batch_previous_covariates, batch_previous_treatments, batch_current_covariates, batch_S)
            batch_current_covariates = batch_current_covariates.reshape(-1, factor_model.num_covariates).float()
            treatment_targets = batch_target_treatments.reshape(-1, factor_model.num_treatments).float()
            outcomes = batch_outcomes.reshape(-1, 1).float()
            confounders = confounders.reshape(-1, factor_model.num_confounders).float()
            loss_a = factor_model.term_a(torch.cat([batch_current_covariates, confounders, S], dim=-1), treatment_targets)
            loss_b = factor_model.term_b(torch.cat([batch_current_covariates, confounders, treatment_targets], dim=-1), outcomes)
            loss_S = factor_model.term_S(torch.cat([batch_current_covariates, confounders, treatment_targets, S], dim=-1), torch.cat([batch_current_covariates, confounders, treatment_targets, outcomes], dim=-1))
            
            # 正则化项
            l2_reg = None
            for W in factor_model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
            
            lambda_reg = 0.1  # 正则化强度
            train_loss = -loss_a - loss_b + args.alpha * loss_S + lambda_reg * l2_reg
            
            # 在optimizer.zero_grad() 和 train_loss.backward() 之后添加梯度监控的代码
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(factor_model.parameters(), max_norm=10)
            # 监控梯度
            max_grad_norm = 0
            for name, param in factor_model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)  # 计算2范数
                    max_grad_norm = max(max_grad_norm, param_norm)
            print(f"Max grad norm: {max_grad_norm.item():.4f}")

            optimizer.step()
            train_losses.append(train_loss.item())


        factor_train_loss = np.mean(train_losses)

        factor_model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_previous_covariates, batch_previous_treatments, batch_current_covariates,
                 batch_target_treatments, batch_outcomes,batch_S) in factor_model.gen_epoch(dataset_val,args):

                # 注意这里的变量名已经根据您的要求进行了更改，重复上述逻辑处理验证集
                confounder_pred_treatments, confounders, S= factor_model(batch_previous_covariates, batch_previous_treatments, batch_current_covariates, batch_S)
                batch_current_covariates = batch_current_covariates.reshape(-1, factor_model.num_covariates).float()
                treatment_targets = batch_target_treatments.reshape(-1, factor_model.num_treatments).float()
                outcomes = batch_outcomes.reshape(-1, 1).float()
                confounders = confounders.reshape(-1, factor_model.num_confounders).float()
                loss_a = factor_model.term_a(torch.cat([batch_current_covariates, confounders,S], dim=-1), treatment_targets)
                loss_b = factor_model.term_b(torch.cat([batch_current_covariates, confounders, treatment_targets], dim=-1), outcomes)
                loss_S = factor_model.term_S(torch.cat([batch_current_covariates, confounders, treatment_targets,S], dim=-1), torch.cat([batch_current_covariates, confounders, treatment_targets, outcomes], dim=-1))
                val_loss =  -loss_a-loss_b + args.alpha *loss_S
                val_losses.append(val_loss)

            factor_val_loss = np.mean([loss.item() for loss in val_losses])

        print(f"Epoch {epoch+1} ---- Train Loss: {factor_train_loss:.5f}, Val Loss: {factor_val_loss:.5f}")
