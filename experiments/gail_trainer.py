from retro_branching.learners import GAILLearner, TreeLearner
from retro_branching.agents import GAILAgent
from retro_branching.networks import BipartiteGCN
from retro_branching.environments import EcoleBranching
from retro_branching.utils import GraphDataset, seed_stochastic_modules_globally, gen_co_name
from retro_branching.loss_functions import CrossEntropy, JensenShannonDistance, KullbackLeiblerDivergence, BinaryCrossEntropyWithLogits, BinaryCrossEntropy, MeanSquaredError

import ecole
import torch 
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
hydra.HYDRA_FULL_ERROR = 1


@hydra.main(config_path='configs', config_name='gail.yaml')
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
    #value_network = BipartiteGCN(device=cfg.experiment.device, **cfg.network)
    # here need modify networks
    actor_network=BipartiteGCN(device=cfg.experiment.device, **cfg.network.actor)
    critic_network=BipartiteGCN(device=cfg.experiment.device, **cfg.network.critic)
    discriminator_network=BipartiteGCN(device=cfg.experiment.device, **cfg.network.discriminator)
    print(f'Initialised network.')

    # initialise imitation agent
    agent = GAILAgent(cfg.experiment.device, actor_network,critic_network,discriminator_network,**cfg.agent)
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

    # init training and validaton data loaders
    experience_data = GraphDataset(sample_files)
    experience_loader = torch_geometric.data.DataLoader(experience_data, batch_size=32, shuffle=True) # defaule batch_size=32
    print('Initialised training and validation data loaders.')

    if 'path_to_instances' in cfg.instances:
        instances = ecole.instance.FileGenerator(cfg.instances.path_to_instances, sampling_mode=cfg.instances.sampling_mode)
    else:
        if cfg.instances.co_class == 'set_covering':
            instances = ecole.instance.SetCoverGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'combinatorial_auction':
            instances = ecole.instance.CombinatorialAuctionGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'capacitated_facility_location':
            instances = ecole.instance.CapacitatedFacilityLocationGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'maximum_independent_set':
            instances = ecole.instance.IndependentSetGenerator(**cfg.instances.co_class_kwargs)
        else:
            raise Exception(f'Unrecognised co_class {cfg.instances.co_class}')
    print(f'Initialised instance generator.')
        
    # initialise branch-and-bound environment
    env = EcoleBranching(observation_function=cfg.environment.observation_function,
                         information_function=cfg.environment.information_function,
                         reward_function=cfg.environment.reward_function,
                         scip_params=cfg.environment.scip_params)
    print(f'Initialised environment.')
    if cfg.experiment.use_tree:
        learner = TreeLearner(agent,
                    env,
                    experience_loader,
                    instances,
                    ecole_seed=cfg.experiment.seed,
                    **cfg.learner)
    else:
        learner = GAILLearner(agent,
            env,
            experience_loader,
            instances,
            ecole_seed=cfg.experiment.seed,
            **cfg.learner)
    print(f'Initialised learner with params {learner.epochs_log}. Will save to {learner.path_to_save}')

    # train agent
    print('Training imitation agent...')
    learner.train(cfg.experiment.num_epochs)

if __name__ == '__main__':
    run()
