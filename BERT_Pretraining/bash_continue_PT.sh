#!/bin/bash
#SBATCH --partition main
#SBATCH --time=7-00:00:00      # time (D-H:MM:SS)
#SBATCH --job-name continue_pt
#SBATCH --mail-type=NONE
#SBATCH --gpus=rtx_3090:1              # Number of GPUs (per node)
#SBATCH --mem=60G
#SBATCH --cpus-per-task=6


### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"


module load anaconda
module load cuda/11.3
source activate FT_Bert
python "Continue_PT.py" --epochs 100 --model_name "pt_from_scratch" --model_path "Final_Model.pkl" --tokenizer_path "Tokenizer.pkl"
