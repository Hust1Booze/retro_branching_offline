from retro_branching.learners import SupervisedLearner
from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads,GPTConfig ,GPT
from retro_branching.utils import GraphDataset,StateActionReturnDataset, StateActionReturnDataset_Test,seed_stochastic_modules_globally, gen_co_name
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

    #train_dataset = StateActionReturnDataset(cfg.OtherConifg.data_path, cfg.OtherConifg.context_length*3)

    # limit by memory, about can load 3000+ epochs, load 10 epochs for test code 
    train_dataset = StateActionReturnDataset_Test(cfg.OtherConifg.data_path, cfg.OtherConifg.context_length*3,max_epochs=10)

    

    # mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
    #                 n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
    mconf = GPTConfig(cfg.GPTConfig.vocab_size, train_dataset.block_size, cfg.network,
                n_layer=cfg.GPTConfig.n_layer,  n_head=cfg.GPTConfig.n_head, n_embd=cfg.GPTConfig.n_embd, model_type=cfg.GPTConfig.model_type, max_timestep=max(train_dataset.timesteps),max_pad_size =cfg.TrainerConfig.max_pad_size)
    
    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    # epochs = args.epochs
    # tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
    #                     lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
    #                     num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))

    tconf = TrainerConfig(max_epochs=cfg.TrainerConfig.max_epochs, batch_size=cfg.TrainerConfig.batch_size, learning_rate=cfg.TrainerConfig.learning_rate,
                        lr_decay=cfg.TrainerConfig.lr_decay, warmup_tokens=cfg.TrainerConfig.warmup_tokens, final_tokens=2*train_dataset.len()* cfg.OtherConifg.context_length*3, # len(train_dataset)
                        num_workers=cfg.TrainerConfig.num_workers, seed=cfg.TrainerConfig.seed, model_type=cfg.TrainerConfig.model_type, 
                        game=cfg.TrainerConfig.game, max_timestep=max(train_dataset.timesteps),ckpt_path = cfg.TrainerConfig.ckpt_path,
                        observation_function=cfg.ValidConfig.observation_function,information_function=cfg.ValidConfig.information_function,
                        reward_function=cfg.ValidConfig.reward_function,scip_params=cfg.ValidConfig.scip_params)
    
    trainer = Trainer(model, train_dataset, None, tconf)

    trainer.train()

    # load the model and eval without train
    #model_state_dict_path = '/home/liutf/code/retro_branching_offline/dt_models/dt_2024-05-23-13-14-26.pt'
    #trainer.eval_with_check_point(model_state_dict_path)



if __name__ == '__main__':
    run()
