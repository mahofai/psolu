#!/bin/bash
#SBATCH -J bayes
#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:1           # Allocate 1 GPU
#SBATCH --mem-per-cpu=96000MB
#SBATCH -o bayes.log
#SBATCH -e bayes.error
#SBATCH -p fastaf2
#SBATCH --time=48:00:00

echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NODELIST=$SLURM_NODELIST"

date
echo "Starting job ..."
python /user/mahaohui/autoML/git/psolu/autoMM.py --lr "0.2 0.3 0.4 0.5 0.6" --mode manual --searcher random

echo "ending job"

date
