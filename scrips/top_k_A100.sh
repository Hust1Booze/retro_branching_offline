#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a100
#SBATCH -J retro_branch_top
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --qos=a100

source activate retro_branch_dt

root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 使用date命令生成带日期时间的字符串
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 构建带有日期时间的文件名
filename="./dt_out/dt_logs/train_dt_${current_time}.txt"

python experiments/dqn_trainer.py --config-name=custom_A100.yaml | tee ${filename}


