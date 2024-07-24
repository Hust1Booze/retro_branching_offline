#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a100
#SBATCH -J gail_trainer
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --qos=a100

source activate retro_branch_dt

root_dir=$(pwd)


python experiments/gail_trainer.py --config-path=configs --config-name=gail.yaml  experiment.path_to_load_imitation_data='/lab/shiyh_lab/12332470/datas' learner.gail_strength=0

