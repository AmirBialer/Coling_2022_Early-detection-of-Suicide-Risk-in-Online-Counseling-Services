#!/bin/bash
#SBATCH --partition main
#SBATCH --time=7-00:00:00      # time (D-H:MM:SS)
#SBATCH --job-name pt_scratch
#SBATCH --mail-type=NONE
#SBATCH --gpus=rtx_3090:1 
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16


### Print some data to output file ###
echo `date` 
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
#

module load anaconda
module load cuda/11.3
source activate venv
python "PT_Bert_From_Scratch.py"