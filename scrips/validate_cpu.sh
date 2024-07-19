#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=cpu
#SBATCH -J validator
#SBATCH -n 40                 # 总核数 40
#SBATCH --ntasks-per-node=40   # 每节点核数
#SBATCH --qos=cpu     


# this shell use CPU and just validate one checkpoint
source activate retro_branch_dt

root_dir=$(pwd)
root_dir='/lab/shiyh_lab/12332470/datas'
echo "root_dir:"${root_dir}

# 使用date命令生成带日期时间的字符串
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 构建带有日期时间的文件名
filename="il_data__${current_time}.txt"

# checkpoint_path
path=/lab/shiyh_lab/12332470/code/retro_branching_offline/outputs/2024-07-19/02-47-16/gail_learner/gail_gnn/gail_gnn_0/checkpoint_52

python experiments/validator.py --config-path=configs --config-name=validator.yaml \
      experiment.agent_name=gail \
      experiment.path_to_load_agent=${path} \
      experiment.path_to_load_instances=/lab/shiyh_lab/12332470/code/retro_branching_offline/retro_branching_paper_validation_instances  \
      experiment.path_to_save=${path} \
      experiment.device=cpu

