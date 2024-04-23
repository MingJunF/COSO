'''
Title: Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders
Authors: Ioana Bica, Ahmed M. Alaa, Mihaela van der Schaar
International Conference on Machine Learning (ICML) 2020

Last Updated Date: July 20th 2020
Code Author: Ioana Bica (ioana.bica95@gmail.com)
'''

from scipy.special import expit
import numpy as np
import pandas as pd
class AutoregressiveSimulation:
    def __init__(self, gamma, num_simulated_hidden_confounders):
        self.num_covariates = 5
        self.num_confounders = num_simulated_hidden_confounders
        self.num_treatments = 1
        self.p = 5

        self.gamma_a = 0.5
        self.gamma_y = 0.5

        self.covariates_coefficients = dict()
        self.covariates_coefficients['treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_covariates, self.num_treatments), treatment_coefficients=True)

        self.covariates_coefficients['covariates'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_covariates, self.num_covariates), variables_coefficients=True)

        self.confounders_coefficients = dict()
        self.confounders_coefficients['treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_confounders, self.num_treatments))
        self.confounders_coefficients['confounders'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_confounders, self.num_confounders), variables_coefficients=True)
        
        self.confounders_coefficients_y = dict()
        self.confounders_coefficients_y['treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_confounders, self.num_treatments))
        self.confounders_coefficients_y['confounders_y'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_confounders, self.num_confounders), variables_coefficients=True)
        self.outcome_coefficients = np.array([np.random.normal(0, 1) for _ in range(self.num_confounders + self.num_covariates)])
        self.treatment_coefficients = self.generate_treatment_coefficients()


    def generate_treatment_coefficients(self):
        treatment_coefficients = np.zeros(shape=(self.num_treatments, self.num_covariates + self.num_confounders))
        for treatment in range(self.num_treatments):
            treatment_coefficients[treatment][treatment] = 1-self.gamma_a
            treatment_coefficients[treatment][self.num_covariates] = self.gamma_a

        return treatment_coefficients


    def generate_coefficients(self, p, matrix_shape, variables_coefficients=False, treatment_coefficients=False):
        coefficients = []
        for i in range(p):
            if (variables_coefficients):
                diag_elements = [np.random.normal(1.0 - (i+1) * 0.2, 0.2) for _ in range(matrix_shape[0])]
                timestep_coefficients = np.diag(diag_elements)

            elif (treatment_coefficients):
                diag_elements = [np.random.normal(0, 0.5)  for _ in range(matrix_shape[1])]
                timestep_coefficients = np.diag(diag_elements)
            else:
                timestep_coefficients = np.random.normal(0, 0.5, size=matrix_shape[1])

            normalized_coefficients = timestep_coefficients / p
            coefficients.append(normalized_coefficients)

        return coefficients

    def generate_treatment_assignments_single_timestep(self, p, history):
        confounders_history = history['confounders']
        covariates_history = history['covariates']

        history_length = len(covariates_history)
        if (history_length < p):
            p = history_length

        average_covariates = np.zeros(shape=len(covariates_history[-1]))
        avearge_confounders = np.zeros(shape=len(confounders_history[-1]))
        for index in range(p):
            average_covariates = average_covariates + covariates_history[history_length - index - 1]
            avearge_confounders = avearge_confounders + confounders_history[history_length - index - 1]

        all_variables = np.concatenate((average_covariates, avearge_confounders)).T

        treatment_assignment = np.zeros(shape=(self.num_treatments,))
        for index in range(self.num_treatments):
            aux_normal = 30 * np.dot(all_variables, self.treatment_coefficients[index])
            treatment_assignment[index] = np.random.binomial(1, expit(aux_normal))

        return treatment_assignment

    def generate_covariates_single_timestep(self, p, history):
        treatments_history = history['treatments']
        covariates_history = history['covariates']

        past_treatment_coefficients = self.covariates_coefficients['treatments']
        past_covariates_coefficients = self.covariates_coefficients['covariates']

        history_length = len(covariates_history)
        if (history_length < p):
            p = history_length

        treatments_sum = np.zeros(shape=(self.num_covariates,))
        covariates_sum = np.zeros(shape=(self.num_covariates,))
        for index in range(p):
            treatments_sum += np.matmul(past_treatment_coefficients[index],
                                        treatments_history[history_length - index - 1])

            covariates_sum += np.matmul(past_covariates_coefficients[index],
                                        covariates_history[history_length - index - 1])

        noise = np.random.normal(0, 0.01, size=(self.num_covariates))

        x_t = treatments_sum + covariates_sum + noise
        x_t = np.clip(x_t, -1, 1)

        return x_t

    def generate_confounders_single_timestep(self, p, history):
        treatments_history = history['treatments']
        confounders_history = history['confounders']
        past_treatment_coefficients = self.confounders_coefficients['treatments']
        past_confounders_coefficients = self.confounders_coefficients['confounders']
        history_length = len(confounders_history)
        if (history_length < p):
            p = history_length

        treatments_sum = np.zeros(shape=(self.num_confounders,))
        confounders_sum = np.zeros(shape=(self.num_confounders,))
        for index in range(p):
            treatments_sum += np.matmul(past_treatment_coefficients[index],
                                        treatments_history[history_length - index - 1])
            confounders_sum += np.matmul(past_confounders_coefficients[index],
                                         confounders_history[history_length - index - 1])
        noise = np.random.normal(0, 0.01, size=(self.num_confounders))

        z_t = treatments_sum + confounders_sum + noise
        z_t = np.clip(z_t, -1, 1)

        return z_t
    def generate_confounders_y_single_timestep(self, p, history):
        treatments_history = history['treatments']
        confounders_history_y = history['confounders_y']
        past_treatment_coefficients = self.confounders_coefficients['treatments']
        past_confounders_coefficients_y = self.confounders_coefficients_y['confounders_y']
        history_length = len(confounders_history_y)
        if (history_length < p):
            p = history_length

        treatments_sum = np.zeros(shape=(self.num_confounders,))
        confounders_sum_y = np.zeros(shape=(self.num_confounders,))
        for index in range(p):
            treatments_sum += np.matmul(past_treatment_coefficients[index],
                                        treatments_history[history_length - index - 1])
            confounders_sum_y += np.matmul(past_confounders_coefficients_y[index],
                                         confounders_history_y[history_length - index - 1])
        noise = np.random.normal(0, 0.01, size=(self.num_confounders))

        y_t = treatments_sum + confounders_sum_y + noise
        y_t = np.clip(y_t, -1, 1)

        return y_t

    def generate_data_single_patient(self, timesteps):

        x_0 = np.random.normal(0, 2, size=(self.num_covariates,))
        z_0 = np.random.normal(0, 2, size=(self.num_confounders,))
        y_0 = np.random.normal(0, 2, size=(self.num_confounders,))
        a_0 = np.zeros(shape=(self.num_treatments,))

        history = dict()
        history['covariates'] = [x_0]
        history['confounders'] = [z_0]
        history['treatments'] = [a_0]
        history['confounders_y'] = [y_0]
        for t in range(timesteps):
            x_t = self.generate_covariates_single_timestep(self.p, history)
            z_t = self.generate_confounders_single_timestep(self.p, history)
            y_t = self.generate_confounders_y_single_timestep(self.p, history)
            history['covariates'].append(x_t)
            history['confounders'].append(z_t)
            history['confounders_y'].append(y_t)
            a_t = self.generate_treatment_assignments_single_timestep(self.p, history)

            history['treatments'].append(a_t)

        return np.array(history['covariates']), np.array(history['confounders']), np.array(history['treatments']),np.array(history['confounders_y'])

    def generate_dict_dataset(self, num_patients, timesteps, p):
        dataset = dict()
        for patient in range(num_patients):
            covariates_history, confounders_history, treatments_history = self.generate_data_single_patient(timesteps,
                                                                                                            p)
            dataset[patient] = dict()
            dataset[patient]['previous_covariates'] = np.array(covariates_history[0:timesteps - 1])
            dataset[patient]['previous_treatments'] = np.array(treatments_history[0:timesteps - 1])
            dataset[patient]['covariates'] = np.array(covariates_history[1:timesteps])
            dataset[patient]['confounders'] = np.array(confounders_history[1:timesteps])
            dataset[patient]['treatments'] = np.array(treatments_history[1:timesteps])
            dataset[patient]['confounders_y'] = np.array(confounders_history[1:timesteps])
        return dataset
    
    def generate_dataset(self, num_patients, max_timesteps):
        dataset = dict()

        dataset['previous_covariates'] = []
        dataset['previous_treatments'] = []
        dataset['covariates'] = []
        dataset['treatments'] = []
        dataset['sequence_length'] = []
        dataset['outcomes'] = []

        for patient in range(num_patients):
            timesteps = np.random.randint(int(max_timesteps) - 10, int(max_timesteps), 1)[0]
            covariates_history, confounders_history, treatments_history, confounders_history_y = self.generate_data_single_patient(timesteps + 1)

            # 合并covariates和confounders历史，使混杂因子作为共变量的一部分
            combined_history = np.concatenate((confounders_history, covariates_history,confounders_history_y), axis=-1)
            if timesteps - 1 < max_timesteps:
                padding_length = max_timesteps - timesteps
                previous_covariates = np.vstack((combined_history[1:timesteps], np.full((padding_length, self.num_covariates + self.num_confounders*2), np.nan)))
                previous_treatments = np.vstack((treatments_history[1:timesteps], np.full((padding_length, self.num_treatments), np.nan)))
                covariates = np.vstack((combined_history[1:timesteps], np.full((padding_length, self.num_covariates + self.num_confounders*2), np.nan)))
                treatments = np.vstack((treatments_history[1:timesteps], np.full((padding_length, self.num_treatments), np.nan)))

            else:

                previous_covariates = combined_history[1:timesteps]
                previous_treatments = treatments_history[1:timesteps]
                covariates = combined_history[1:timesteps]
                treatments = treatments_history[1:timesteps]
        
            outcomes = self.gamma_y * np.mean(confounders_history_y[2:timesteps + 1], axis=-1) +  self.gamma_y * np.mean(covariates_history[2:timesteps + 1], axis=-1)
            outcomes = outcomes[:, np.newaxis]
            outcomes = np.vstack((outcomes, np.full((padding_length, 1), np.nan)))
            dataset['previous_covariates'].append(previous_covariates)
            dataset['previous_treatments'].append(previous_treatments)
            dataset['covariates'].append(covariates)
            dataset['treatments'].append(treatments)
            dataset['sequence_length'].append(timesteps)
            dataset['outcomes'].append(outcomes)
        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])
        # 计算 scaling parameters
        mean_outcome = np.mean(dataset['outcomes'][~np.isnan(dataset['outcomes'])])
        std_outcome = np.std(dataset['outcomes'][~np.isnan(dataset['outcomes'])])
        scaling_params = {'output_means': mean_outcome, 'output_stds': std_outcome}
        dataset['outcomes_scaled'] = (dataset['outcomes'] - mean_outcome) / std_outcome  # 标准化结果
        static_features = np.random.rand(num_patients, 1)  # Assuming a single static feature for simplicity
        return  dataset['treatments'], dataset['outcomes_scaled'], dataset['covariates'], static_features, dataset['outcomes'], scaling_params, dataset['covariates']







