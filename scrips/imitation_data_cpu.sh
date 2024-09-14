#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=cpu
#SBATCH -J gen_data
#SBATCH -n 40                 # 总核数 40
#SBATCH --ntasks-per-node=40   # 每节点核数
#SBATCH --qos=cpu     

source activate retro_branch_dt

root_dir=$(pwd)
root_dir='/lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/datasets'
echo "root_dir:"${root_dir}


python experiments/gen_imitation_data.py --config-path=configs --config-name=gen_imitation_data.yaml experiment.path_to_save=${root_dir} instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000
#python experiments/gen_imitation_data.py --config-path=configs --config-name=gen_imitation_data.yaml experiment.path_to_save=${root_dir} instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000
python experiments/gen_imitation_data.py --config-path=configs --config-name=gen_imitation_data_cl.yaml experiment.path_to_save=${root_dir} instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000
#python experiments/gen_imitation_data.py --config-path=configs --config-name=gen_imitation_data_cl.yaml experiment.path_to_save=${root_dir} instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000

