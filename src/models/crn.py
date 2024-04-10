from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
import math
from typing import Union
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
from copy import deepcopy
from pytorch_lightning import Trainer
import ray
from ray import tune
from ray import ray_constants

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.time_varying_model import BRCausalModel
from src.models.utils import grad_reverse, BRTreatmentOutcomeHead, AlphaRise
from src.models.utils_lstm import VariationalLSTM


logger = logging.getLogger(__name__)


class CRN(BRCausalModel):
    """
    Pytorch-Lightning implementation of Counterfactual Recurrent Network (CRN)
    (https://arxiv.org/abs/2002.04083, https://github.com/ioanabica/Counterfactual-Recurrent-Network)
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = {'encoder', 'decoder'}

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

    def _init_specific(self, sub_args: DictConfig):
        # Encoder/decoder-specific parameters
        try:
            self.br_size = sub_args.br_size  # balanced representation size
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            self.num_layer = sub_args.num_layer
            self.batch_size = sub_args.batch_size
            # Pytorch model init
            if self.seq_hidden_units is None or self.br_size is None or self.fc_hidden_units is None or self.dropout_rate is None:
                raise MissingMandatoryValue()

            self.lstm = VariationalLSTM(self.input_size, self.seq_hidden_units, self.num_layer, self.dropout_rate)

            self.br_treatment_outcome_head = BRTreatmentOutcomeHead(self.seq_hidden_units, self.br_size, self.fc_hidden_units,
                                                                    self.dim_treatments, self.dim_outcome, self.alpha,
                                                                    self.update_alpha, self.balancing)

        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Used for hyperparameter tuning and model reinitialisation
        :param model_args: Sub DictConfig, with encoder/decoder parameters
        :param new_args: New hyperparameters
        :param input_size: Input size of the model
        :param model_type: Submodel specification
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args['learning_rate']
        sub_args.batch_size = new_args['batch_size']
        if 'seq_hidden_units' in new_args:  # Only relevant for encoder
            sub_args.seq_hidden_units = int(input_size * new_args['seq_hidden_units'])
        sub_args.br_size = int(input_size * new_args['br_size'])
        sub_args.fc_hidden_units = int(sub_args.br_size * new_args['fc_hidden_units'])
        sub_args.dropout_rate = new_args['dropout_rate']
        sub_args.num_layer = new_args['num_layer']

    def build_br(self, prev_treatments, vitals_or_prev_outputs, static_features, init_states=None):
        x = torch.cat((prev_treatments, vitals_or_prev_outputs), dim=-1)
        x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        x = self.lstm(x, init_states=init_states)
        br = self.br_treatment_outcome_head.build_br(x)
        return br




class CRNEncoder(CRN):

    model_type = 'encoder'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None, has_vitals: bool = None, bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.input_size = self.dim_treatments + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome if self.autoregressive else 0
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self._init_specific(args.model.encoder)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        # Datasets normalisation etc.
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_encoder:
            self.dataset_collection.process_data_encoder()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        prev_treatments = batch['prev_treatments']
        vitals_or_prev_outputs = []
        vitals_or_prev_outputs.append(batch['vitals']) if self.has_vitals else None
        vitals_or_prev_outputs.append(batch['prev_outputs']) if self.autoregressive else None
        vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)

        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        init_states = None  # None for encoder
        #logger.info(f"Dimension of treatment:{curr_treatments}")
        #logger.info(f"Dimension of outcome:{batch['prev_outputs'].shape}")

        br = self.build_br(prev_treatments, vitals_or_prev_outputs, static_features, init_states)
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)

        return treatment_pred, outcome_pred, br


class CRNDecoder(CRN):

    model_type = 'decoder'

    def __init__(self, args: DictConfig,
                 encoder: CRNEncoder = None,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 encoder_r_size: int = None, autoregressive: bool = None, has_vitals: bool = None, bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        self.input_size = self.dim_treatments + self.dim_static_features + self.dim_outcome
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self.encoder = encoder
        args.model.decoder.seq_hidden_units = self.encoder.br_size if encoder is not None else encoder_r_size
        self._init_specific(args.model.decoder)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        # Datasets normalisation etc.
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_decoder:
            self.dataset_collection.process_data_decoder(self.encoder)
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        prev_treatments = batch['prev_treatments']
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        init_states = batch['init_state']

        br = self.build_br(prev_treatments, prev_outputs, static_features, init_states)
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)

        return treatment_pred, outcome_pred, br
class COSO(CRN):
    model_type = 'COSO'
    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self._init_specific(args.model.coso)
    def trainable_init_h_confounder(self):
        h0 = torch.zeros(1, self.batch_size,self.seq_hidden_units)
        c0 = torch.zeros(1,self.batch_size, self.seq_hidden_units)
        z0 = torch.zeros(self.batch_size,1 ,self.num_confounders)
        trainable_h0 = nn.Parameter(h0, requires_grad=True)
        trainable_c0 = nn.Parameter(c0, requires_grad=True)
        trainable_z0 = nn.Parameter(z0, requires_grad=True)
        return trainable_h0, trainable_c0, trainable_z0


    def forward(self, previous_covariates, previous_treatments, current_covariates):
        batch_size = previous_covariates.size(0)
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