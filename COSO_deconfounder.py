'''
Title: Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders
Authors: Ioana Bica, Ahmed M. Alaa, Mihaela van der Schaar
International Conference on Machine Learning (ICML) 2020

Last Updated Date: July 20th 2020
Code Author: Ioana Bica (ioana.bica95@gmail.com)
'''
import logging
import numpy as np
import os
import shutil
import torch
from sklearn.model_selection import ShuffleSplit

from utils.evaluation_utils import write_results_to_file
from factor_model import FactorModel
from rmsn.script_rnn_fit import rnn_fit
from rmsn.script_rnn_test import rnn_test
from rmsn.script_propensity_generation import propensity_generation
from COSO_Factor_model import COSOFactorModel
from train_model import train_model
from CBD.MBs.pc_simple import pc_simple
def train_factor_model(dataset_train, dataset_val, dataset, num_confounders, hyperparams_file,
                       b_hyperparameter_optimisation):
    _, length, num_covariates = dataset_train['covariates'].shape
    num_treatments = dataset_train['treatments'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_confounders': num_confounders,
              'max_sequence_length': length,
              'num_epochs': 100}

    hyperparams = dict()
    num_simulations = 50
    best_validation_loss = 100
    if b_hyperparameter_optimisation and False:
        logging.info("Performing hyperparameter optimization")
        for simulation in range(num_simulations):
            logging.info("Simulation {} out of {}".format(simulation + 1, num_simulations))

            hyperparams['rnn_hidden_units'] = np.random.choice([32, 64, 128, 256])
            hyperparams['fc_hidden_units'] = np.random.choice([32, 64, 128])
            hyperparams['learning_rate'] = np.random.choice([0.01, 0.001, 0.0001])
            hyperparams['batch_size'] = np.random.choice([64, 128, 256])
            hyperparams['rnn_keep_prob'] = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9])

            logging.info("Current hyperparams used for training \n {}".format(hyperparams))
            model = FactorModel(params, hyperparams)
            model.train(dataset_train, dataset_val)
            validation_loss = model.eval_network(dataset_val)

            if (validation_loss < best_validation_loss):
                logging.info(
                    "Updating best validation loss | Previous best validation loss: {} | Current best validation loss: {}".format(
                        best_validation_loss, validation_loss))
                best_validation_loss = validation_loss
                best_hyperparams = hyperparams.copy()

            logging.info("Best hyperparams: \n {}".format(best_hyperparams))

        write_results_to_file(hyperparams_file, best_hyperparams)

    else:
        best_hyperparams = {
            'rnn_hidden_units': 128,
            'fc_hidden_units': 128,
            'learning_rate': 0.01,
            'batch_size': 64,
            'rnn_keep_prob': 0.7}

    model = FactorModel(params, best_hyperparams)
    model.train(dataset_train, dataset_val)
    predicted_confounders = model.compute_hidden_confounders(dataset)

    return predicted_confounders


def get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders):
    if use_predicted_confounders:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments',
                        'predicted_confounders', 'outcomes','S']
    else:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments', 'outcomes','S']

    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    for key in dataset_keys:
        dataset_train[key] = dataset[key][train_index, :, :]
        dataset_val[key] = dataset[key][val_index, :, :]
        dataset_test[key] = dataset[key][test_index, :, :]

    _, length, num_covariates = dataset_train['covariates'].shape

    key = 'sequence_length'
    dataset_train[key] = dataset[key][train_index]
    dataset_val[key] = dataset[key][val_index]
    dataset_test[key] = dataset[key][test_index]

    dataset_map = dict()

    dataset_map['num_time_steps'] = length
    dataset_map['training_data'] = dataset_train
    dataset_map['validation_data'] = dataset_val
    dataset_map['test_data'] = dataset_test

    return dataset_map


def train_rmsn(dataset_map, model_name, b_use_predicted_confounders):
    model_name = model_name + '_use_confounders_' + str(b_use_predicted_confounders)
    MODEL_ROOT = os.path.join('results', model_name)

    if not os.path.exists(MODEL_ROOT):
        os.mkdir(MODEL_ROOT)
        print("Directory ", MODEL_ROOT, " Created ")
    else:
        # Need to delete previously saved model.
        shutil.rmtree(MODEL_ROOT)
        os.mkdir(MODEL_ROOT)
        print("Directory ", MODEL_ROOT, " Created ")

    rnn_fit(dataset_map=dataset_map, networks_to_train='propensity_networks', MODEL_ROOT=MODEL_ROOT,
            b_use_predicted_confounders=b_use_predicted_confounders)

    propensity_generation(dataset_map=dataset_map, MODEL_ROOT=MODEL_ROOT,
                          b_use_predicted_confounders=b_use_predicted_confounders)

    rnn_fit(networks_to_train='encoder', dataset_map=dataset_map, MODEL_ROOT=MODEL_ROOT,
            b_use_predicted_confounders=b_use_predicted_confounders)

    rmsn_mse = rnn_test(dataset_map=dataset_map, MODEL_ROOT=MODEL_ROOT,
                        b_use_predicted_confounders=b_use_predicted_confounders)

    rmse = np.sqrt(np.mean(rmsn_mse)) * 100
    return rmse


def test_time_series_deconfounder(dataset, num_substitute_confounders, exp_name, dataset_with_confounders_filename,
                                  factor_model_hyperparams_file, args, b_hyperparm_tuning=False):
    S = find_S_variable(dataset)
    dataset['S'] = S

    # Splitting the dataset
    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.1, random_state=10)
    train_index, test_index = next(shuffle_split.split(dataset['covariates'][:, :, 0]))

    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.11, random_state=10)
    train_index, val_index = next(shuffle_split.split(dataset['covariates'][train_index, :, 0]))

    dataset_map = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=False)
    dataset_train = dataset_map['training_data']
    dataset_val = dataset_map['validation_data']

    logging.info("Fitting factor model")


    predicted_confounders = train_COSO_factor_model(dataset_train, dataset_val, dataset,train_index, val_index, test_index,
                                                    num_confounders=num_substitute_confounders,
                                                    hyperparams_file=factor_model_hyperparams_file,
                                                    b_hyperparameter_optimisation=b_hyperparm_tuning, args=args)
    predicted_confounders = predicted_confounders.reshape(-1, dataset['covariates'].shape[1], args.num_substitute_hidden_confounders)
    dataset['predicted_confounders'] = predicted_confounders
    write_results_to_file(dataset_with_confounders_filename, dataset)

    dataset_map_COSO = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=True)

    logging.info('Fitting counfounded recurrent marginal structural networks.')
    rmse_without_COSO = train_rmsn(dataset_map_COSO, 'rmsn_' + str(exp_name), b_use_predicted_confounders=False)

    logging.info(
        'Fitting deconfounded (D_Z = {}) recurrent marginal structural networks.'.format(num_substitute_confounders))
    rmse_with_COSO = train_rmsn(dataset_map_COSO, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True)
    print("Outcome model RMSE when trained WITHOUT the COSO.")
    print(rmse_without_COSO)

    print("Outcome model RMSE when trained WITH the substitutes for the COSO.")
    print(rmse_with_COSO)
    predicted_confounders = train_factor_model(dataset_train, dataset_val,
                                            dataset,
                                            num_confounders=num_substitute_confounders,
                                            b_hyperparameter_optimisation=b_hyperparm_tuning,
                                            hyperparams_file=factor_model_hyperparams_file)
    predicted_confounders = predicted_confounders.reshape(-1, dataset['covariates'].shape[1], args.num_substitute_hidden_confounders)
    dataset['predicted_confounders'] = predicted_confounders
    write_results_to_file(dataset_with_confounders_filename, dataset)
    

    dataset_map_deconfounder = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=True)

    logging.info('Fitting counfounded recurrent marginal structural networks.')
    rmse_without_confounders = train_rmsn(dataset_map_deconfounder, 'rmsn_' + str(exp_name), b_use_predicted_confounders=False)

    logging.info(
        'Fitting deconfounded (D_Z = {}) recurrent marginal structural networks.'.format(num_substitute_confounders))
    rmse_with_confounders = train_rmsn(dataset_map_deconfounder, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True)
    print('rmsn_' + str(exp_name))
    print("Outcome model RMSE when trained WITHOUT the hidden_confounder.")
    print(rmse_without_confounders)

    print("Outcome model RMSE when trained WITH the substitutes for the hidden_confounder.")
    print(rmse_with_confounders)

    




    
def train_COSO_factor_model(dataset_train, dataset_val, dataset,train_index, val_index, test_index, num_confounders, hyperparams_file,
                            b_hyperparameter_optimisation, args):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    _, length, num_covariates = dataset_train['covariates'].shape
    num_treatments = dataset_train['treatments'].shape[-1]
    hyperparams_rmse_pairs = []
    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_confounders': num_confounders,
              'max_sequence_length': length,
              'num_epochs': 100,
              'num_outcomes': 1,
              'num_S': 1,
              }

    best_rmse = np.inf  # Initialize best RMSE to infinity
    best_hyperparams = {}  # Initialize best hyperparameters

    if b_hyperparameter_optimisation:
        num_simulations = 30
        logging.info("Performing hyperparameter optimization")

        for simulation in range(num_simulations):
            logging.info(f"Simulation {simulation + 1} out of {num_simulations}")

            hyperparams = {
                'lstm_hidden_units': np.random.choice([32, 64, 128, 256]),
                'fc_hidden_units': np.random.choice([32, 64, 128]),
                'learning_rate': np.random.choice([0.01, 0.001, 0.0001]),
                'batch_size': np.random.choice([32, 64, 128]),
            }

            logging.info(f"Current hyperparams used for training: {hyperparams}")
            model = COSOFactorModel(params, hyperparams, args)
            try:    train_model(model, dataset_train, dataset_val, args)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory during training. Trying to empty cache.")
                    torch.cuda.empty_cache()
            predicted_confounders = model.compute_hidden_confounders(dataset, args)
            dataset['predicted_confounders'] = predicted_confounders.reshape(-1, length, num_confounders)

            # Assuming get_dataset_splits and train_rmsn functions are implemented correctly
            dataset_map_COSO = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=True)
            rmse_with_COSO = train_rmsn(dataset_map_COSO, 'rmsn_temp', b_use_predicted_confounders=True)
            hyperparams_rmse_pairs.append((hyperparams, rmse_with_COSO))
            if rmse_with_COSO < best_rmse:
                logging.info(f"Updating best RMSE: {best_rmse} -> {rmse_with_COSO}")
                best_rmse = rmse_with_COSO
                best_hyperparams = hyperparams.copy()

        write_results_to_file(hyperparams_file, best_hyperparams)

    else:
        best_hyperparams = {
            'lstm_hidden_units': 128,
            'fc_hidden_units': 128,
            'learning_rate': 0.01,
            'batch_size': 64}

    # Train the model with the best hyperparameters after hyperparameter optimization
    best_hyperparams_converted = [( {k: int(v) if isinstance(v, np.int32) else v for k, v in pair[0].items()}, pair[1] ) for pair in hyperparams_rmse_pairs]

    save_best_hyperparams_to_file(best_hyperparams_converted, directory="./hyperparams")
    model = COSOFactorModel(params, best_hyperparams, args)
    train_model(model, dataset_train, dataset_val, args)
    predicted_confounders = model.compute_hidden_confounders(dataset, args)
    return predicted_confounders

def check_time_correspondence(dataset):
    num_patients = len(dataset['previous_covariates'])
    correspondence = True

    for patient in range(num_patients):
        # 获取实际的时间序列长度
        actual_length = dataset['sequence_length'][patient]
        
        # 对比previous_covariates和covariates
        prev_covs = dataset['previous_covariates'][patient][:actual_length-1]  # 考虑实际长度
        covs = dataset['covariates'][patient][1:actual_length]  # 从第二个时间步开始，基于实际长度
        
        if not np.array_equal(prev_covs, covs):
            print(f"Mismatch in covariates for patient {patient}")
            correspondence = False
            break

        # 对比previous_treatments和treatments
        prev_treats = dataset['previous_treatments'][patient][:actual_length-1]  # 考虑实际长度
        treats = dataset['treatments'][patient][1:actual_length]  # 从第二个时间步开始，基于实际长度

        if not np.array_equal(prev_treats, treats):
            print(f"Mismatch in treatments for patient {patient}")
            correspondence = False
            break

    if correspondence:
        print("All previous_covariates and previous_treatments correctly precede covariates and treatments respectively.")
    else:
        print("There is a mismatch in the dataset related to time steps.")
def find_S_variable(dataset):
        num_patients, timesteps, num_covariates = dataset['previous_covariates'].shape
        _, _, num_treatments = dataset['previous_treatments'].shape
        most_relevant_var_for_each_patient = np.full((num_patients, 1), np.nan)

        for patient in range(num_patients):
            all_data_for_analysis = []
            for time in range(1, timesteps):
                features = np.hstack([
                    dataset['previous_covariates'][patient, time - 1, :].flatten(),
                    dataset['previous_treatments'][patient, time - 1, :].flatten(),
                ])
                current_treatment = dataset['treatments'][patient, time, :].flatten()
                current_outcome = dataset['outcomes'][patient, time, :].flatten()
                data_for_analysis = np.hstack([features, current_treatment, current_outcome]).reshape(1, -1)
                all_data_for_analysis.append(data_for_analysis)

            if all_data_for_analysis:
                concatenated_data = np.concatenate(all_data_for_analysis, axis=0)
                _, _, treatment_pvals = pc_simple(concatenated_data, target=len(features), alpha=0.05, isdiscrete=True)
                _, _, outcome_pvals = pc_simple(concatenated_data, target=len(features) + len(current_treatment), alpha=0.05, isdiscrete=True)

                treatment_related_vars = {var for var, pval in treatment_pvals.items() if pval <= 0.05}
                outcome_unrelated_vars = {var for var, pval in outcome_pvals.items() if pval > 0.05}

                relevant_vars = treatment_related_vars.difference(outcome_unrelated_vars)

                if relevant_vars:
                    min_pval = float('inf')
                    most_relevant_var = None
                    for var in relevant_vars:
                        if var in treatment_pvals and treatment_pvals[var] < min_pval:
                            min_pval = treatment_pvals[var]
                            most_relevant_var = var
                    most_relevant_var_for_each_patient[patient, 0] = most_relevant_var

        # 分析最频繁出现的变量
        valid_vars = most_relevant_var_for_each_patient[~np.isnan(most_relevant_var_for_each_patient)]
        if len(valid_vars) > 0:
            (unique_vars, counts) = np.unique(valid_vars, return_counts=True)
            most_frequent_var_index = np.argmax(counts)
            most_frequent_var = unique_vars[most_frequent_var_index]
        else:
            return None  # 没有找到有效的变量

        if most_frequent_var is not None:
            most_frequent_var = int(most_frequent_var)

        num_patients, timesteps, _ = dataset['previous_covariates'].shape
        # 使用np.full创建一个数组，每个元素都填充索引值
        S_variable_set = np.full((num_patients, timesteps + 1, 1), most_frequent_var, dtype=np.float32)
        return S_variable_set



import json
import os

def save_best_hyperparams_to_file(best_hyperparams, directory, filename="best_hyperparams.json"):
    """
    将最佳超参数保存到JSON文件中。
    
    参数:
    - best_hyperparams: 最佳超参数的字典。
    - directory: 要保存文件的目录路径。
    - filename: 保存的文件名，默认为'best_hyperparams.json'。
    """
    # 确保目录存在，如果不存在，则创建它
    os.makedirs(directory, exist_ok=True)
    
    # 构建完整的文件路径
    filepath = os.path.join(directory, filename)
    
    # 将最佳超参数写入JSON文件
    with open(filepath, 'w') as file:
        json.dump(best_hyperparams, file, indent=4)  # 使用indent参数美化输出
    
    print(f"最佳超参数已保存到文件：{filepath}")