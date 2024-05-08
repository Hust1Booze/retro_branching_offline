#!/bin/bash
echo "bash test"
root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 使用date命令生成带日期时间的字符串
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 构建带有日期时间的文件名
filename="train_offline_${current_time}.txt"

python experiments/dqn_trainer.py --config-path=configs --config-name=retro_il.yaml experiment.device=cuda:0 learner.path_to_save=. instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230 | tee "$filename"

