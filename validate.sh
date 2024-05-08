#!/bin/bash
echo "bash test"
root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 使用date命令生成带日期时间的字符串
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 构建带有日期时间的文件名
filename="validate_${current_time}.txt"

total=/home/liutf/code/retro_branching/scratch/datasets/offline_retro/dqn_learner/dqn_gnn/dqn_gnn_3/
for item in "$total"/*; do
  if [ -d "$item" ] && [[ $item == *"checkpoint"* ]]; then  # 如果是目录
    #echo "Found directory: $item"
    path="${item}/"
    #echo ${path}
    if [ ! -d "$path/rl_validator" ]; then
	echo "begin validate $path"
    	python experiments/validator.py --config-path=configs --config-name=validator.yaml environment.observation_function=43_var_and_sb_features environment.scip_params=gasse_2019 instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000 experiment.agent_name=offline_retro experiment.path_to_load_agent=${path} experiment.path_to_load_instances=${root_dir}/retro_branching_paper_validation_instances experiment.path_to_save=${path} experiment.device=cuda:0 | tee "$filename"
    fi    
  fi
done
# For offline_retro
#path_to_agent=/home/liutf/code/retro_branching/outputs/2024-04-03/18-58-18/dqn_learner/dqn_gnn/dqn_gnn_0/checkpoint_58/
#python experiments/validator.py --config-path=configs --config-name=validator.yaml environment.observation_function=43_var_and_sb_features environment.scip_params=gasse_2019 instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230 experiment.agent_name=offline_retro experiment.path_to_load_agent=${path_to_agent} experiment.path_to_load_instances=${root_dir}/retro_branching_paper_validation_instances experiment.path_to_save=${path_to_agent} experiment.device=cuda:0 | tee "$filename"

# For retro validate
#path_to_agent=/home/liutf/code/retro_branching/outputs/2024-04-01/19-16-28/dqn_learner/dqn_gnn/dqn_gnn_0/checkpoint_58/
#python experiments/validator.py --config-path=configs --config-name=validator.yaml environment.observation_function=43_var_features environment.scip_params=gasse_2019 instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230 experiment.agent_name=retro experiment.path_to_load_agent=${path_to_agent} experiment.path_to_load_instances=${root_dir}/retro_branching_paper_validation_instances experiment.path_to_save=${path_to_agent} experiment.device=cuda:0 | tee "$filename"

# Strong branch
#python experiments/validator.py --config-path=configs --config-name=validator.yaml environment.scip_params=gasse_2019 instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000 experiment.agent_name=strong_branching experiment.path_to_load_instances=${root_dir}/retro_branching_paper_validation_instances/ experiment.path_to_save=${root_dir}/retro_branching_paper_validation_agents/ experiment.device=cpu
