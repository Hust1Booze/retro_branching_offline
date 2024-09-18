#!/bin/bash

# use example : ./my_script.sh /your/path/here
root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 使用date命令生成带日期时间的字符串
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 构建带有日期时间的文件名
filename="validate_${current_time}.txt"

# Get the path parameter
total_path=$1/

#指定checkpoint
if [ -n "$2" ]; then
  echo "one checkpoint"
  path="${total_path}/checkpoint_$2"
  echo "begin validate $path"
  python experiments/validator.py --config-path=configs --config-name=validator.yaml \
  environment.observation_function=default  \
  instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000 \
  experiment.agent_name=gail experiment.path_to_load_agent=${path} \
  experiment.path_to_load_instances=${root_dir}/retro_branching_paper_validation_instances \
  experiment.path_to_save=${path}  | tee "${path}/${filename}"
  exit 1
fi
# 遍历文件夹，对每个checkpoint validate
#for item in "$total_path"/*; do
#倒序遍历，从最新的checkpoint开始
for item in $(find "$total_path" -mindepth 1 -maxdepth 1 -type d -name "checkpoint_*" | sort -V -r); do
  if [ -d "$item" ] && [[ $item == *"checkpoint"* ]]; then  # 如果是目录
    #echo "Found directory: $item"
    path="${item}/"
    #echo ${path}
    if [ ! -d "$path/rl_validator" ]; then
	echo "begin validate $path"
    	python experiments/validator.py --config-path=configs --config-name=validator.yaml \
      environment.observation_function=default  \
      instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000 \
      experiment.agent_name=il experiment.path_to_load_agent=${path} \
      experiment.path_to_load_instances=${root_dir}/retro_branching_paper_validation_instances \
      experiment.path_to_save=${path}  | tee "${path}/${filename}"
    fi    
  fi
done
