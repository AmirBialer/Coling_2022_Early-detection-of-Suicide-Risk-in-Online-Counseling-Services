#!/bin/bash
#SBATCH --partition main
#SBATCH --time=7-00:00:00      # time (D-H:MM:SS)
#SBATCH --job-name FT
#SBATCH --mail-type=NONE
##SBATCH --gpus=1              # Number of GPUs (per node)
#SBATCH --mem=60G


### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST


module load anaconda
module load cuda/11.3
source activate Bert_37
python "FT_Model.py" --model_path "100.0epochs_Model.pkl" --tail_or_head "keep_head" --epochs 20 --label "gsr" --exp "reg"
