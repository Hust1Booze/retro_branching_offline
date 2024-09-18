from retro_branching.learners import SupervisedLearner
from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads, BipartiteGCN_Cl
from retro_branching.utils import GraphDataset, TripleGraphDataset, seed_stochastic_modules_globally, gen_co_name
from retro_branching.loss_functions import CrossEntropy, JensenShannonDistance, KullbackLeiblerDivergence, BinaryCrossEntropyWithLogits, BinaryCrossEntropy, MeanSquaredError

import torch_geometric 
import pathlib
import glob
import numpy as np
import os
import random
import torch

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
hydra.HYDRA_FULL_ERROR = 1


@hydra.main(config_path='configs', config_name='cl.yaml')
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
    if cfg.learner.loss_function != 'infoNCE':
        if cfg.experiment.path_to_load_agent is not '':
            path = cfg.experiment.path_to_load_agent + f'/{gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)}/{cfg.experiment.agent_name}/'
            config = path + 'config.json'
            agent = BipartiteGCN_Cl(device=cfg.experiment.device, config=config, name=cfg.experiment.agent_name)
            for network_name, network in agent.get_networks().items():
                if network is not None:
                    try:
                        # see if network saved under same var as 'network_name'
                        agent.__dict__[network_name].load_state_dict(torch.load(path+f'/{network_name}_params.pkl', map_location=cfg.experiment.device))
                    except KeyError:
                        # network saved under generic 'network' var (as in Agent class)
                        agent.__dict__['network'].load_state_dict(torch.load(path+f'/{network_name}_params.pkl', map_location=cfg.experiment.device))
                else:
                    print(f'{network_name} is None.')
            print(f'Train il with cl pre-train model in {config}')
        else:
            print('Train il')
            agent = BipartiteGCN_Cl(device=cfg.experiment.device, **cfg.network) # None 'add'
    else:
        print('Train CL')
        agent = BipartiteGCN_Cl(device=cfg.experiment.device, **cfg.network) # None 'add'
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
    print(f'there is total {len(files)} samples')
    sample_files = files[:cfg.experiment.num_samples]
    files = [] # clear memory
    train_files = sample_files[:int(0.83*len(sample_files))]
    valid_files = sample_files[int(0.83*len(sample_files)):]

    if cfg.learner.loss_function == 'infoNCE':
        train_data = TripleGraphDataset(train_files)
        train_loader = torch_geometric.data.DataLoader(train_data, batch_size=32, shuffle=True)
        valid_data = TripleGraphDataset(valid_files)
        valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=512, shuffle=False) 
    else:  
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
    elif cfg.learner.loss_function == 'infoNCE':
        loss_function = None
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
                                checkpoint_frequency=cfg.learner.checkpoint_log_frequency,
                                save_logits_and_target=True,
                                path_to_save=cfg.experiment.path_to_save,
                                name=cfg.learner.name)
    print(f'Initialised learner with params {learner.epochs_log}. Will save to {learner.path_to_save}')

    # train agent
    print('Training imitation agent...')
    learner.train(cfg.experiment.num_epochs)

if __name__ == '__main__':
    run()