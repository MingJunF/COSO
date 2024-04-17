@echo off

:: Set the working directory
cd /d D:\Mingjun\COSO

:: Activate the conda environment
call conda activate CT

:: Set necessary environment variables
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0

:: Run train_rmsn.py script 5 times
echo Starting runs for train_rmsn.py
for /L %%i in (1,1,5) do (
    echo Running iteration %%i of train_rmsn.py
	python runnables/train_rmsn.py -m +dataset=mimic3_real +backbone=rmsn +backbone/rmsn_hparams/mimic3_real=diastolic_blood_pressure exp.seed=10,100,1000,10000,100000
)

:: Run train_multi.py script 5 times
echo Starting runs for train_multi.py
for /L %%i in (1,1,5) do (
    echo Running iteration %%i of train_multi.py
    python runnables/train_multi.py -m +dataset=mimic3_real +backbone=ct +backbone/ct_hparams/mimic3_real=diastolic_blood_pressure exp.seed=10,100,1000,10000,100000
)

:: Run train_coso.py script 5 times
echo Starting runs for train_crn.py
for /L %%i in (1,1,5) do (
    echo Running iteration %%i of train_crn.py
    python runnables/train_enc_dec.py -m +dataset=mimic3_real +backbone=crn +backbone/crn_hparams/mimic3_real=diastolic_blood_pressure exp.seed=10,100,1000,10000,100000
)


echo All runs complete
pause
