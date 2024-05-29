#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a100
#SBATCH -J gen_data
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --qos=a100

source activate retro_branch_dt

root_dir=$(pwd)
root_dir='/lab/shiyh_lab/12332470/datas'
echo "root_dir:"${root_dir}

# 使用date命令生成带日期时间的字符串
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 构建带有日期时间的文件名
filename="il_data__${current_time}.txt"

python experiments/gen_imitation_data.py --config-path=configs --config-name=gen_imitation_data.yaml experiment.path_to_save=${root_dir} instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230
