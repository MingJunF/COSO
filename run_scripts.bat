@echo off

:: Set the working directory
cd /d D:\Mingjun\COSO

:: Activate the conda environment
call conda activate CT

:: Set necessary environment variables
set PYTHONPATH=%CD%
set CUDA_VISIBLE_DEVICES=0
python runnables/train_multi.py -m +dataset=mimic3_real +backbone=ct +backbone/ct_hparams/mimic3_real='0','1','2','3','diastolic_blood_pressure' exp.seed=10
echo All runs complete

