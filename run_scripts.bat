@echo off

:: Set the working directory
cd /d D:\Mingjun\COSO

:: Activate the conda environment
call conda activate CT

:: Set necessary environment variables
set PYTHONPATH=%CD%
set CUDA_VISIBLE_DEVICES=0

:: Run train_rmsn.py script 5 times
echo Starting runs for train_rmsn.py
for /L %%i in (1,1,1) do (
    echo Running iteration %%i of train_rmsn.py
	python runnables/train_coso.py -m +dataset=cancer_sim +backbone=coso +backbone/coso_hparams/cancer_sim_domain_conf='0'
)



echo All runs complete
pause
