#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=cpu
#SBATCH -J il_validate
#SBATCH -n 40                 # 总核数 40
#SBATCH --ntasks-per-node=40   # 每节点核数
#SBATCH --qos=cpu     

source activate retro_branch_dt

./scrips/validate.sh /lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/outputs/scratch_models/il_20/gnn/gnn_0/
./scrips/validate.sh /lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/outputs/scratch_models/il_30/gnn/gnn_0/
./scrips/validate.sh /lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/outputs/scratch_models/il_40/gnn/gnn_0/
./scrips/validate.sh /lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/outputs/scratch_models/il_50/gnn/gnn_0/
./scrips/validate.sh /lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/outputs/scratch_models/il_60/gnn/gnn_0/
./scrips/validate.sh /lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/outputs/scratch_models/il_70/gnn/gnn_0/
./scrips/validate.sh /lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/outputs/scratch_models/il_80/gnn/gnn_0/
./scrips/validate.sh /lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/outputs/scratch_models/il_90/gnn/gnn_0/
./scrips/validate.sh /lab/shiyh_lab/12332470/code/contrastive/retro_branching_offline/outputs/scratch_models/il_100/gnn/gnn_0/