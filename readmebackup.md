CFPnet
==============================
Counterfactual Prediction from Longitudinal Data with Time-Varying Latent Confounders

To maintain code reusability, our work is based on CausalTransformer and follows the same directory structure. URL: https://github.com/Valentyn1997/CausalTransformer.

### Installations
To start, please download following Python libraries:
1. [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) - deep learning models
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [MlFlow](https://mlflow.org/) - experiments tracking
All the other required libraries have been saved in requirements.txt with the required versions. For simple installation, please try:
```console
pip3 install -r requirements.txt
```

## MlFlow Setup / Connection
To start an experiment, an MLflow server is required.
To start the experiment server, run:

'mlflow ui'

The relevant experiment results will be available in your local browser at http://localhost:5000 or the link returned after running mlflow ui.

## Experiments

Main training script is universal for different models and datasets. For details on mandatory arguments - see the main configuration file `config/config.yaml` and other files in `configs/` folder.

The basic structure for running the experiments are as following:
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 runnables/train_<training-type>.py +dataset=<dataset> +backbone=<backbone> exp.seed=<number>
```

### Backbones (baselines)
One needs to choose a backbone and then fill the specific hyperparameters by +backbone=<> (they are left blank in the configs):
- [CounterFactual Prediction network](CFPnet): `runnables/train_CFPnet.py  +backbone=CFPnet`
- [Causal Transformer](https://arxiv.org/abs/2204.07258) (CT): `runnables/train_multi.py  +backbone=ct`
- [Recurrent Marginal Structural Networks](https://papers.nips.cc/paper/2018/hash/56e6a93212e4482d99c84a639d254b67-Abstract.html) (RMSNs): `runnables/train_rmsn.py +backbone=rmsn`
- [Counterfactual Recurrent Network](https://arxiv.org/abs/2002.04083) (CRN): `runnables/train_enc_dec.py +backbone=crn`
- [G-Net](https://proceedings.mlr.press/v158/li21a/li21a.pdf): `runnables/train_gnet.py  +backbone=gnet`


The used hyperparameters in experiment are saved (for each model and dataset). You can access them via: +backbone/<backbone>_hparams/Simulated_data=<Relevant file name> or +backbone/<backbone>_hparams/mimic3_real=diastolic_blood_pressure. For comparability, CFPnet kept the same hyperparameters as CRN in the encoder-decoder part.

For CFPnet, CT, EDCT, and CT, several adversarial balancing objectives are available:
- counterfactual domain confusion loss (originally in CT, but can be used for CRN): `exp.balancing=domain_confusion`
- gradient reversal (originally in CRN, but can be used for all the methods): `exp.balancing=grad_reverse`
To train a decoder (for CFPnet, CRN and RMSNs), use the flag `model.train_decoder=True`.
To perform a manual hyperparameter tuning use the flags `model.<sub_model>.tune_hparams=True`, and then see `model.<sub_model>.hparams_grid`. Use `model.<sub_model>.tune_range` to specify the number of trials for random search.

### Datasets
One needs to specify a dataset / dataset generator (and some additional parameters):
- Synthetic dataset: `+dataset=Simulated_data`
- MIMIC III Real-world dataset: `+dataset=mimic3_real`

Before running MIMIC III experiments, place MIMIC-III-extract dataset ([all_hourly_data.h5](https://github.com/MLforHealth/MIMIC_Extract)) to `data/processed/`

To run experiments with CFPnet on MIMIC-III for 5 different seeds, please try the following or simply click 'run_scripts': 

```console
set PYTHONPATH=%python%
set CUDA_VISIBLE_DEVICES=0
python runnables/train_CFPnet.py -m +dataset=mimic3_real +backbone=CFPnet +backbone/CFPnet_hparams/mimic3_real=diastolic_blood_pressure exp.seed=10,100,1000,10000,100000
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
