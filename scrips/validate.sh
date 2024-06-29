#!/bin/bash

# use example : ./my_script.sh /your/path/here
root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 使用date命令生成带日期时间的字符串
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 构建带有日期时间的文件名
filename="validate_${current_time}.txt"

# Get the path parameter
total_path=$1

# 遍历文件夹，对每个checkpoint validate
for item in "$total_path"/*; do
  if [ -d "$item" ] && [[ $item == *"checkpoint"* ]]; then  # 如果是目录
    #echo "Found directory: $item"
    path="${item}/"
    #echo ${path}
    if [ ! -d "$path/rl_validator" ]; then
	echo "begin validate $path"
    	python experiments/validator.py --config-path=configs --config-name=validator.yaml environment.observation_function=43_var_and_sb_features  instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000 experiment.agent_name=offline_retro experiment.path_to_load_agent=${path} experiment.path_to_load_instances=${root_dir}/retro_branching_paper_validation_instances experiment.path_to_save=${path}  | tee "$filename"
    fi    
  fi
done
