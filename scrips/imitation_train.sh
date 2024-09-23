#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a100
#SBATCH -J il
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --qos=a100

source activate retro_branch_dt

echo "bash test"
root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 使用date命令生成带日期时间的字符串
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 构建带有日期时间的文件名
filename="train_il_${current_time}.txt"

#python experiments/il.py --config-path=configs --config-name=il.yaml experiment.path_to_save=${root_dir} instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000

python experiments/imitation_trainer.py   --config-path=configs --config-name=il.yaml experiment.path_to_load_imitation_data='/lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/datasets'  experiment.num_samples=24000 learner.name='il_20'
python experiments/imitation_trainer.py   --config-path=configs --config-name=il.yaml experiment.path_to_load_imitation_data='/lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/datasets'  experiment.num_samples=36000 learner.name='il_30'
python experiments/imitation_trainer.py   --config-path=configs --config-name=il.yaml experiment.path_to_load_imitation_data='/lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/datasets'  experiment.num_samples=48000 learner.name='il_40'
python experiments/imitation_trainer.py   --config-path=configs --config-name=il.yaml experiment.path_to_load_imitation_data='/lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/datasets'  experiment.num_samples=60000 learner.name='il_50'
python experiments/imitation_trainer.py   --config-path=configs --config-name=il.yaml experiment.path_to_load_imitation_data='/lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/datasets'  experiment.num_samples=72000 learner.name='il_60'
python experiments/imitation_trainer.py   --config-path=configs --config-name=il.yaml experiment.path_to_load_imitation_data='/lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/datasets'  experiment.num_samples=84000 learner.name='il_70'
python experiments/imitation_trainer.py   --config-path=configs --config-name=il.yaml experiment.path_to_load_imitation_data='/lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/datasets'  experiment.num_samples=96000 learner.name='il_80'
python experiments/imitation_trainer.py   --config-path=configs --config-name=il.yaml experiment.path_to_load_imitation_data='/lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/datasets'  experiment.num_samples=108000 learner.name='il_90'
python experiments/imitation_trainer.py   --config-path=configs --config-name=il.yaml experiment.path_to_load_imitation_data='/lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/datasets'  experiment.num_samples=120000 learner.name='il_100'