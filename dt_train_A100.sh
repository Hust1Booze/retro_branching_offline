#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a100
#SBATCH -J retro_branch_dt
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --qos=a100

source activate retro_branch_dt
#python experiments/imitation_trainer.py --config-path=configs --config-name=il.yaml experiment.device=cuda:0 experiment.path_to_save=./model_output experiment.path_to_load_imitation_data=/home/lab06/shiyh_lab/cse12332470/code/retro_branching  instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230

#python experiments/dqn_trainer.py --config-path=configs --config-name=retro.yaml experiment.device=cuda:0 learner.path_to_save=. instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000

root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 使用date命令生成带日期时间的字符串
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 构建带有日期时间的文件名
filename="./dt_out/dt_logs/train_dt_${current_time}.txt"

#python experiments/imitation_trainer.py --config-path=configs --config-name=il.yaml experiment.device=cuda:0 experiment.path_to_save=./model_output experiment.path_to_load_imitation_data=${root_dir}  instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230

python experiments/dt_trainer.py | tee ${filename}


