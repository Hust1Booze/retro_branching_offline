from retro_branching.learners import SupervisedLearner
from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads,GPTConfig ,GPT
from retro_branching.utils import GraphDataset,StateActionReturnDataset, seed_stochastic_modules_globally, gen_co_name
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

    # initialise imitation agent
    graph_net = BipartiteGCNNoHeads(device=cfg.experiment.device, **cfg.network) # None 'add'
    graph_net.to(cfg.experiment.device)
    graph_net.train() # turn on train mode
    print('Initialised imitation agent.')


    train_dataset = StateActionReturnDataset(cfg.OtherConifg.data_path, cfg.OtherConifg.context_length*3)

    # mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
    #                 n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                n_layer=cfg.GPTConfig.n_layer,  n_head=cfg.GPTConfig.n_head, n_embd=cfg.GPTConfig.n_embd, model_type=cfg.GPTConfig.model_type, max_timestep=max(train_dataset.timesteps))
    
    model = GPT(mconf,graph_net)

    # initialize a trainer instance and kick off training
    # epochs = args.epochs
    # tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
    #                     lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
    #                     num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))

    tconf = TrainerConfig(max_epochs=cfg.TrainerConfig.max_epochs, batch_size=cfg.TrainerConfig.batch_size, learning_rate=cfg.TrainerConfig.learning_rate,
                        lr_decay=cfg.TrainerConfig.lr_decay, warmup_tokens=cfg.TrainerConfig.warmup_tokens, final_tokens=2*len(train_dataset)* cfg.OtherConifg.context_length*3,
                        num_workers=cfg.TrainerConfig.num_workers, seed=cfg.TrainerConfig.seed, model_type=cfg.TrainerConfig.model_type, game=cfg.TrainerConfig.game, max_timestep=max(train_dataset.timesteps))
    trainer = Trainer(model, train_dataset, None, tconf)

    trainer.train()

    # initialise imitation agent
    agent = BipartiteGCN(device=cfg.experiment.device, **cfg.network) # None 'add'
    agent.to(cfg.experiment.device)
    agent.train() # turn on train mode
    print('Initialised imitation agent.')

    # get paths to labelled training and validation data
    folder_name = 'samples_1' # 'aggregated_samples' 'samples_1'
    path = cfg.experiment.path_to_load_imitation_data + f'/{cfg.experiment.branching}/{cfg.instances.co_class}/max_steps_{cfg.experiment.max_steps}/{gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)}/samples/{folder_name}/'

    print(f'Loading imitation data from {path}...')
    if not os.path.isdir(path):
        raise Exception(f'Path {path} does not exist')
    files = np.array(glob.glob(path+'*.pkl'))
    sample_files = files[:cfg.experiment.num_samples]
    files = [] # clear memory
    train_files = sample_files[:int(0.83*len(sample_files))]
    valid_files = sample_files[int(0.83*len(sample_files)):]

    # init training and validaton data loaders
    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=512, shuffle=False)
    print('Initialised training and validation data loaders.')

    # init learner
    if cfg.learner.loss_function == 'cross_entropy':
        loss_function = CrossEntropy()
    elif cfg.learner.loss_function == 'mean_squared_error':
        loss_function = MeanSquaredError()
    elif cfg.learner.loss_function == 'jensen_shannon_distance':
        loss_function = JensenShannonDistance()
    elif cfg.learner.loss_function == 'kullback_leibler_divergence':
        loss_function = KullbackLeiblerDivergence()
    else:
        raise Exception(f'Unrecognised loss_function {cfg.learner.loss_function}')
    learner = SupervisedLearner(agent=agent,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                imitation_target=cfg.learner.imitation_target, # 'expert_scores' 'expert_score' 'expert_actions' 'expert_bipartite_ranking'
                                loss_function=loss_function, # MeanSquaredError() CrossEntropy() JensenShannonDistance() KullbackLeiblerDivergence()
                                lr=cfg.learner.lr,
                                bipartite_ranking_alpha=0.5,
                                epoch_log_frequency=cfg.learner.epoch_log_frequency,
                                checkpoint_frequency=cfg.learner.checkpoint_frequency,
                                save_logits_and_target=True,
                                path_to_save=cfg.experiment.path_to_save,
                                name=cfg.learner.name)
    print(f'Initialised learner with params {learner.epochs_log}. Will save to {learner.path_to_save}')

    # train agent
    print('Training imitation agent...')
    print('Training imitation2 agent...')
    learner.train(cfg.experiment.num_epochs)

if __name__ == '__main__':
    run()
