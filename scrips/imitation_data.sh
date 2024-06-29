#!/bin/bash
echo "bash test"
root_dir=$(pwd)
root_dir='/home/liutf/datas'
echo "root_dir:"${root_dir}

# 使用date命令生成带日期时间的字符串
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 构建带有日期时间的文件名
filename="train_il_${current_time}.txt"

python experiments/gen_imitation_data.py --config-path=configs --config-name=gen_imitation_data.yaml experiment.path_to_save=${root_dir} instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230

#python experiments/imitation_trainer.py --config-path=configs --config-name=il.yaml experiment.device=cuda:0 experiment.path_to_save=./model_output experiment.path_to_load_imitation_data=${root_dir}  instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230
