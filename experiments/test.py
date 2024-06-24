from retro_branching.utils import gen_co_name, ExploreThenStrongBranch, PureStrongBranch, seed_stochastic_modules_globally
from retro_branching.scip_params import gasse_2019_scip_params, default_scip_params

import ecole

import gzip
import pickle
import numpy as np
from pathlib import Path
import time
import os
import glob
import random

import ray
import psutil
NUM_CPUS = psutil.cpu_count(logical=False)
try:
    ray.init(num_cpus=NUM_CPUS)
except RuntimeError:
    # already initialised ray in script calling dcn sim, no need to init again
    pass


import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
hydra.HYDRA_FULL_ERROR = 1


# def run_sampler(co_class, branching, nrows, ncols, max_steps=None, instance=None):
def run_sampler(co_class, co_class_kwargs, branching, max_steps=None, instance=None):
    '''
    Args:
        branching (str): Branching scheme to use. Must be one of 'explore_then_strong_branch',
            'pure_strong_branch'
        max_steps (None, int): If not None, will terminate episode after max_steps.
    '''
    if instance is None:
        # N.B. Need to init instances and env here since ecole objects are not
        # serialisable and ray requires all args passed to it to be serialisable
        if co_class == 'set_covering':
            instances = ecole.instance.SetCoverGenerator(**co_class_kwargs)
        elif co_class == 'combinatorial_auction':
            instances = ecole.instance.CombinatorialAuctionGenerator(**co_class_kwargs)
        elif co_class == 'capacitated_facility_location':
            instances = ecole.instance.CapacitatedFacilityLocationGenerator(**co_class_kwargs)
        elif co_class == 'maximum_independent_set':
            instances = ecole.instance.IndependentSetGenerator(**co_class_kwargs)
        else:
            raise Exception(f'Unrecognised co_class {co_class}')
        instance = next(instances)
    else:
        # already have an instance
        if type(instance) == str:
            # load instance from path
            instance = ecole.scip.Model.from_file(instance)

    # scip_params = default_scip_params
    scip_params = gasse_2019_scip_params

    information_function=({
            'num_nodes': ecole.reward.NNodes(),
            'lp_iterations': ecole.reward.LpIterations().cumsum(),
            'solving_time': ecole.reward.SolvingTime().cumsum(),
            'primal_integral': ecole.reward.PrimalIntegral().cumsum(),
            'dual_integral': ecole.reward.DualIntegral(),
            'primal_dual_integral': ecole.reward.PrimalDualIntegral(),
            'IsDone': ecole.reward.IsDone(),
        })

    if branching == 'explore_then_strong_branch':
        env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=0.05), 
                                                                ecole.observation.NodeBipartite()), 
                                          scip_params=scip_params)
    elif branching == 'pure_strong_branch':
        env = ecole.environment.Branching(observation_function=(PureStrongBranch(), 
                                                                ecole.observation.NodeBipartite()), 
                                                                information_function = information_function,
                                                                scip_params=scip_params)
    else:
        raise Exception('Unrecognised branching {}'.format(branching))

    observation, action_set, _, done, info = env.reset(instance)
    print(info)
    data_to_save = []
    t = 0
    while not done:
        if branching == 'explore_then_strong_branch':
            # only save samples if they are coming from the expert (strong branching)
            (scores, save_samples), node_observation = observation
        elif branching == 'pure_strong_branch':
            # always save samples since always using strong branching
            save_samples = True
            scores, node_observation = observation
        else:
            raise Exception('Unrecognised branching {}'.format(branching))

        action = action_set[scores[action_set].argmax()]

        if save_samples:
            data = [node_observation, action, action_set, scores]
            data_to_save.append(data)

        observation, action_set, _, done, info= env.step(action)
        print(info)
        t += 1
        if max_steps is not None:
            if t >= max_steps:
                # stop episode
                break
    
    return data_to_save



@hydra.main(config_path='configs', config_name='gen_imitation_data.yaml')
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
    for i in range(10):
        print('*'*10)
        run_sampler(co_class=cfg.instances.co_class, 
                                co_class_kwargs=cfg.instances.co_class_kwargs,
                                branching=cfg.experiment.branching, 
                                max_steps=cfg.experiment.max_steps,
                                instance=None)

if __name__ == '__main__':
    run()









