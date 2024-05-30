from retro_branching.learners import SupervisedLearner
from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads,GPTConfig ,GPT
from retro_branching.utils import GraphDataset,StateActionReturnDataset, StateActionReturnDataset_Test,Data_loader,seed_stochastic_modules_globally, gen_co_name
from retro_branching.loss_functions import CrossEntropy, JensenShannonDistance, KullbackLeiblerDivergence, BinaryCrossEntropyWithLogits, BinaryCrossEntropy, MeanSquaredError

import torch_geometric 
import pathlib
import glob
import numpy as np
import os
import random


import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
from retro_branching.learners import TrainerConfig, Trainer
hydra.HYDRA_FULL_ERROR = 1


@hydra.main(config_path='configs', config_name='dt.yaml')
def run(cfg: DictConfig):
    # seeding
    if 'seed' not in cfg.experiment:
        cfg.experiment['seed'] = random.randint(0, 10000)
    seed_stochastic_modules_globally(cfg.experiment.seed)

    # print info
    print('\n\n\n')
    print(f'~'*80)
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~'*80)


    # have to add this to reload cfg, if not do this cant merge,dont know why
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))

    # limit by memory, about can load 3000+ epochs, load 10 epochs for test code 
    loader = Data_loader(cfg.DataConfig.data_path,cfg.DataConfig.max_epochs)

    train_dataset = StateActionReturnDataset_Test(loader.train_data,cfg.DataConfig.context_length*3)
    test_dataset = StateActionReturnDataset_Test(loader.test_data,cfg.DataConfig.context_length*3)

    # those configs need load dynamically
    extra_cfg = OmegaConf.create({
        "GPTConfig": {
            "block_size": cfg.DataConfig.context_length*3,
            "max_timestep": max(train_dataset.timesteps),
            "max_pad_size": cfg.TrainerConfig.max_pad_size,
            "graph_net": cfg.network
        },
        "TrainerConfig":{

            "final_tokens": 2*train_dataset.len()* cfg.DataConfig.context_length*3,
            "max_timestep": max(train_dataset.timesteps),
            "observation_function": cfg.ValidConfig.observation_function,
            "information_function": cfg.ValidConfig.information_function,
            "reward_function": cfg.ValidConfig.reward_function,
            "scip_params": cfg.ValidConfig.scip_params
        }
    })

    # 合并配置
    merged_cfg = OmegaConf.merge(cfg, extra_cfg)
  
    model = GPT(merged_cfg.GPTConfig)

    trainer = Trainer(model, train_dataset, test_dataset, merged_cfg.TrainerConfig)

    trainer.train()

    # load the model and eval without train
    #model_state_dict_path = '/home/liutf/code/retro_branching_offline/dt_models/dt_2024-05-23-13-14-26.pt'
    #trainer.eval_with_check_point(model_state_dict_path)



if __name__ == '__main__':
    run()
