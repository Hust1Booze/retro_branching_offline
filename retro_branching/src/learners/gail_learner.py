import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='ray')  # noqa

from retro_branching.src.learners.learner import Learner
from retro_branching.loss_functions import MeanSquaredError
from retro_branching.src.learners.dqn_learner import _reset_env, BipartiteNodeData, extract_state_tensors_from_ecole_obs

import torch
from torch_geometric.data import Batch

import time
from collections import defaultdict
import copy
import numpy as np
import random
import threading

import ray
import pickle
from itertools import zip_longest


def _extract_reward(reward, extrinsic_reward, intrinsic_reward, intrinsic_extrinsic_combiner, episode_stats, multiple_rewards=False, train_predictor=True):
    if multiple_rewards:
        _reward = []
        if isinstance(extrinsic_reward, list):
            for r in extrinsic_reward:
                _reward.append(reward[r])
                episode_stats[f'extrinsic_{r}_R'] += reward[r]
                episode_stats['R'] += reward[r]
        else:
            _reward.append(reward[extrinsic_reward])
            episode_stats[f'extrinsic_{extrinsic_reward}_R'] += reward[extrinsic_reward]
            episode_stats['R'] += reward[extrinsic_reward]
    else:
        _reward = reward[extrinsic_reward]
        episode_stats[f'extrinsic_{extrinsic_reward}_R'] += reward[extrinsic_reward]
        episode_stats['R'] += reward[extrinsic_reward]

    # if intrinsic_reward is not None:
        # if 'subtree' not in extrinsic_reward:
            # intrinsic_reward = intrinsic_reward.extract(env.model, done, train_predictor=train_predictor)
            # if self.intrinsic_extrinsic_combiner == 'add':
                # _reward += intrinsic_reward
            # elif self.intrinsic_extrinsic_combiner == 'list':
                # if not isinstance(_reward, list):
                    # _reward = [_reward]
                # _reward.append(intrinsic_reward)
            # else:
                # raise Exception(f'Unrecognised intrinsic_extrinsic_combiner {self.intrinsic_extrinsic_combiner}')
            # episode_stats[f'intrinsic_{self.intrinsic_reward.name}_R'] += intrinsic_reward
            # episode_stats['R'] += intrinsic_reward
        # else:
            # # can only handle intrinsic rewards when know sub-tree
            # pass

    return _reward, episode_stats

def _init_episode_stats():
    episode_stats = defaultdict(lambda: 0)

    # if self.multiple_rewards:
        # if isinstance(self.agent_reward, list):
            # episode_stats['R'] = {r: 0 for r in self.agent_reward}
        # elif self.intrinsic_reward is not None:
            # if self.intrinsic_extrinsic_combiner == 'list':
                # episode_stats['R'] = {self.agent_reward: 0, self.intrinsic_reward.name: 0}
            # elif self.intrinsic_extrinsic_combiner == 'add':
                # # will combine intrinsic extrinsic rewards into single reward, do not need dict to store separately
                # pass
            # else:
                # raise Exception(f'Not sure how to handle agent_reward={self.agent_reward} intrinsic_reward={self.intrinsic_reward} multiple_rewards={self.multiple_rewards}')
        # else:
            # raise Exception(f'Not sure how to handle agent_reward={self.agent_reward} intrinsic_reward={self.intrinsic_reward} multiple_rewards={self.multiple_rewards}')

    return episode_stats

def reset_env(ecole_seed,
              reproducible_episodes,
              observation_function, 
              information_function,
              reward_function,
              scip_params,
              instances, 
              intrinsic_reward=None,
              max_attempts=500):
    if isinstance(instances, str):
        # single path to instance
        instance = instances
        env, obs, action_set, reward, done, info, instance_before_reset = _reset_env(instance=instance,
                                                                                     observation_function=observation_function,
                                                                                     information_function=information_function,
                                                                                     reward_function=reward_function,
                                                                                     scip_params=scip_params,
                                                                                     ecole_seed=ecole_seed,
                                                                                     reproducible_episodes=reproducible_episodes)
    else:
        # iterable, find instance which is not pre-solved
        num_attempts = 0
        obs = None
        while obs is None:
            instance = next(instances)
            # if intrinsic_reward is not None:
                # if 'subtree' in self.extrinsic_reward:
                    # # can only calc intrinsic reward when retrospectively know sub-tree episode -> do not reset yet
                    # intrinsic_reward = None
                # else:
                    # # calc intrinsic reward at each step in episode -> reset now
                    # intrinsic_reward = self.intrinsic_reward
            # else:
                # intrinsic_reward = None

            env, obs, action_set, reward, done, info, instance_before_reset = _reset_env(instance=instance,
                                                                                         observation_function=observation_function,
                                                                                         information_function=information_function,
                                                                                         reward_function=reward_function,
                                                                                         scip_params=scip_params,
                                                                                         ecole_seed=ecole_seed,
                                                                                         reproducible_episodes=reproducible_episodes)
            num_attempts += 1
            if num_attempts >= max_attempts:
                raise Exception(f'Unable to find instance which is not pre-solved after {num_attempts} attempts.')

    return env, obs, action_set, reward, done, info, instance_before_reset

@ray.remote
def run_parallel_episode(agent,
        extrinsic_reward,
        intrinsic_reward,
        intrinsic_extrinsic_combiner,
        ecole_seed,
        reproducible_episodes,
                         episode_id, 
                         _buffer, 
                         device, 
                         observation_function,
                         information_function,
                         reward_function,
                         scip_params,
                         instances,
                         train_predictor=True,
                         debug_mode=False):
    return _run_episode(agent=agent,
                        extrinsic_reward=extrinsic_reward,
                        intrinsic_reward=intrinsic_reward,
                        intrinsic_extrinsic_combiner=intrinsic_extrinsic_combiner,
                        ecole_seed=ecole_seed,
                                                        reproducible_episodes=reproducible_episodes,
                                                           episode_id=episode_id, 
                                                           _buffer=_buffer,
                                                           device=device,
                                                           observation_function=observation_function,
                                                           information_function=information_function,
                                                           reward_function=reward_function,
                                                           scip_params=scip_params,
                                                           instances=instances,
                                                           debug_mode=debug_mode)

def run_sequential_episode(agent,
        ecole_seed,
        reproducible_episodes,
                         episode_id, 
                         _buffer, 
                         device, 
                         observation_function,
                         information_function,
                         reward_function,
                         scip_params,
                         instances,
                         train_predictor=True,
                         debug_mode=False):
    return _run_episode(agent,
            ecole_seed=ecole_seed,
                        reproducible_episodes=reproducible_episodes,
                           episode_id=episode_id, 
                           _buffer=_buffer,
                           device=device,
                           observation_function=observation_function,
                           information_function=information_function,
                           reward_function=reward_function,
                           scip_params=scip_params,
                           instances=instances,
                           debug_mode=debug_mode)


def _run_episode(agent,
             ecole_seed,
             reproducible_episodes,
             episode_id, 
             _buffer, 
             device, 
             observation_function,
             information_function,
             reward_function,
             scip_params,
             instances,
             train_predictor=True,
             debug_mode=False):
    if debug_mode:
        print(f'\nStarting episode...')

    episode_stats = _init_episode_stats()

    env, obs, action_set, reward, done, info, instance_before_reset = reset_env(ecole_seed=ecole_seed,
                                                                                     reproducible_episodes=reproducible_episodes,
                                                                                     observation_function=observation_function,
                                                                                          information_function=information_function,
                                                                                          reward_function=reward_function,
                                                                                          scip_params=scip_params,
                                                                                          instances=instances)
    if done:
        # pre-solved, do not consider further
        return episode_id, _buffer, episode_stats

    episode_step_counter = 0
    start_t = time.time()
    while not done:
        action_set = action_set.astype(int)

        # update buffer
        state = extract_state_tensors_from_ecole_obs(obs, action_set)
        _buffer.states.append(state)
        _buffer.action_sets.append(action_set)


        # use actor to select action
        action, action_idx = agent.action_select(action_set=action_set, obs=obs)

        _reward = agent.get_expert_reward(obs, action_set, action_idx)

        # step env
        obs, action_set, reward, done, info = env.step(action)
        episode_step_counter += 1

        # update buffer
        _buffer.action_idxs.append(action_idx)
        _buffer.logprobs.append(agent.action_logprob)
        _buffer.rewards.append(_reward)
        _buffer.dones.append(done)

        if debug_mode:
            print(f'step {episode_step_counter} || action: {action} | action_idx: {action_idx} | reward: {_reward} | done: {done}')
    if debug_mode:
        print(f'Finished episode.')
    end_t = time.time()


    episode_stats['num_nodes'] = info['num_nodes']
    episode_stats['num_steps'] = episode_step_counter
    episode_stats['lp_iterations'] = info['lp_iterations']
    episode_stats['num_actor_steps'] = episode_step_counter
    episode_stats['episode_run_time'] = end_t - start_t

    return episode_id, _buffer, episode_stats







class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.action_idxs = []
        self.action_sets = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.states)

class GAILLearner(Learner):
    def __init__(self,
                 agent,
                 env,
                 experience_loader,
                 instances,
                 ecole_seed=0,
                 reproducible_episodes=True,
                 value_function_coeff=0.5,
                 entropy_coeff=0.01,
                 eps_clip=0.2,
                 whiten_rewards=True,
                 lr_actor=1e-4,
                 lr_critic=5e-4,
                 lr_discriminator = 5e-4,
                 critic_loss_function=None,
                 discriminator_loss_function=None,
                 batch_size=32,
                 gradient_accumulation_factor=1,
                 actor_gradient_clipping_clip_value=None,
                 critic_gradient_clipping_clip_value=None,
                 ppo_update_freq=1,
                 ppo_epochs_per_update=1,
                 gamma=0.75,
                 num_workers=None,
                 episode_log_freq=1,
                 epoch_log_freq=1,
                 checkpoint_freq=1,
                 path_to_save=None,
                 use_sqlite_database=False,
                 profile_time=False,
                 debug_mode=False,
                 name='ppo_learner'):
        super(GAILLearner, self).__init__(agent, path_to_save, name)

        self.agent = agent
        self.agent.train()
        self.env = env
        self.experience_loader = experience_loader
        self.ecole_seed = ecole_seed
        self.reproducible_episodes = reproducible_episodes
        self.instances = instances


        self.value_function_coeff = value_function_coeff
        self.entropy_coeff = entropy_coeff
        self.eps_clip = eps_clip
        self.whiten_rewards = whiten_rewards

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_discriminator = lr_discriminator
        # self.optimizer = self.reset_optimizer()
        self.actor_optimizer = torch.optim.Adam(self.agent.actor_network.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.agent.critic_network.parameters(), lr=lr_critic)
        self.discriminator_optimizer = torch.optim.Adam(self.agent.discriminator_network.parameters(), lr=lr_discriminator)

        if critic_loss_function is None:
            self.critic_loss_function = MeanSquaredError()
        else:
            self.critic_loss_function = critic_loss_function

        if discriminator_loss_function is None:
            self.discriminator_loss_function = torch.nn.BCELoss()
        else:
            self.discriminator_loss_function = discriminator_loss_function

        self.batch_size = batch_size
        self.gradient_accumulation_factor = gradient_accumulation_factor
        self.ppo_update_freq = ppo_update_freq
        self.ppo_epochs_per_update = ppo_epochs_per_update
        self.actor_gradient_clipping_clip_value = actor_gradient_clipping_clip_value 
        self.critic_gradient_clipping_clip_value = critic_gradient_clipping_clip_value 

        self.gamma = gamma

        self.num_workers = num_workers
        if self.num_workers is not None:
            ray.init(num_cpus=self.num_workers)

        self.episode_log_freq = episode_log_freq
        self.epoch_log_freq = epoch_log_freq
        self.checkpoint_freq = checkpoint_freq
        if path_to_save is not None:
            self.path_to_save = self.init_save_dir(path=path_to_save, use_sqlite_database=use_sqlite_database)
        else:
            self.path_to_save = None
        self.save_thread = None
        self.use_sqlite_database = use_sqlite_database
        self.name = name

        self.reset_episodes_log()
        self.reset_epochs_log()

        self.profile_time = profile_time
        self.debug_mode = debug_mode

    def _save(self):
        if self.save_thread is not None:
            self.save_thread.join()
        self.save_thread = threading.Thread(target=self.save_checkpoint, 
                                            args=({'episodes_log': copy.deepcopy(self.episodes_log), 'epochs_log': copy.deepcopy(self.epochs_log)}, self.use_sqlite_database,))
        self.save_thread.start()
        if self.use_sqlite_database:
            # reset in-memory logs
            self.reset_episodes_log()
            self.reset_epochs_log()

    def gather_experiences(self):
        episode_to_buffer = {episode_id: RolloutBuffer() for episode_id in range(self.ppo_update_freq)}
        if self.num_workers is None:
            # gather experiences sequentially
            for episode_id in episode_to_buffer.keys():
                _, episode_to_buffer[episode_id], episode_stats = run_sequential_episode(agent=self.agent,
                                                                           ecole_seed=self.ecole_seed,
                                                                           reproducible_episodes=self.reproducible_episodes,
                                                                           episode_id=episode_id,
                                                                           _buffer=episode_to_buffer[episode_id],
                                                                           device=self.agent.device,
                                                                           observation_function=self.env.str_observation_function,
                                                                           information_function=self.env.str_information_function,
                                                                           reward_function=self.env.str_reward_function,
                                                                           scip_params=self.env.str_scip_params,
                                                                           instances=self.instances,
                                                                           debug_mode=self.debug_mode,
                                                                           )
                self.episode_counter += 1
                self.actor_step_counter += episode_stats['episode_step_counter']
                self.update_episodes_log(episode_stats)
                if self.episode_counter % self.episode_log_freq == 0:
                    print(self.get_episode_log_str())
        else:
            # gather experiences in parallel
            start_t = time.time()
            orig_agent_device = copy.deepcopy(self.agent.device)
            self.agent = self.move_agent_to_device(self.agent, 'cpu')
            counter, episode_ids = 0, list(episode_to_buffer.keys())
            num_nodes = 0
            while True:
                result_ids = []
                num_runs_left = int(len(episode_to_buffer.keys()) - counter)
                if num_runs_left == 0:
                    break
                else:
                    num_processes = min(self.num_workers, num_runs_left)
                    for _ in range(num_processes):
                        # episode_id = next(episode_ids)
                        episode_id = random.choice(episode_ids)
                        episode_ids.remove(episode_id)
                        result_ids.append(run_parallel_episode.remote(agent=self.agent,
                            ecole_seed=self.ecole_seed,
                                                                           reproducible_episodes=self.reproducible_episodes,
                                                                           episode_id=episode_id,
                                                                           _buffer=episode_to_buffer[episode_id],
                                                                           device='cpu',
                                                                           observation_function=self.env.str_observation_function,
                                                                           information_function=self.env.str_information_function,
                                                                           reward_function=self.env.str_reward_function,
                                                                           scip_params=self.env.str_scip_params,
                                                                           instances=next(self.instances),
                                                                           debug_mode=self.debug_mode,
                                                                           ))

                    # collect results
                    outputs = ray.get(result_ids)
                    for i in range(len(outputs)):
                        episode_id, _buffer, episode_stats = outputs[i]
                        if len(_buffer) > 0:
                            episode_to_buffer[episode_id] = _buffer
                            self.update_episodes_log(episode_stats)
                            self.episode_counter += 1
                            self.actor_step_counter += episode_stats['episode_step_counter']
                            counter += 1
                            num_nodes += episode_stats['num_nodes']
                        else:
                            # pre-solved, do not count -> need to re-consider this episode
                            episode_ids.append(episode_id)
                            pass
            run_t = round(time.time() - start_t, 3)
            mean_num_nodes = round(num_nodes / len(episode_to_buffer.keys()), 3)
            print(f'Completed {len(episode_to_buffer.keys())} episodes with {self.num_workers} workers in {run_t} s -> Mean # nodes: {mean_num_nodes}')

            # put agent back onto original device
            self.agent = self.move_agent_to_device(self.agent, orig_agent_device)

        return episode_to_buffer


    def train(self, num_ppo_epochs):
        self.num_ppo_epochs = num_ppo_epochs
        
        self.actor_step_counter = 0
        self.episode_counter = 0
        self.train_start_t = time.time()
        print(f'Training {self.agent.name} agent for {num_ppo_epochs} epochs...')
        self.network_epoch_counter = 0
        self.ppo_epoch_counter = 0

        for self.ppo_epoch_counter in range(self.num_ppo_epochs):
            if self.debug_mode:
                print(f'\n>>> PPO epoch {self.ppo_epoch_counter+1} of {self.num_ppo_epochs} <<<')

            if self.path_to_save is not None:
                if self.ppo_epoch_counter % self.checkpoint_freq == 0:
                    self._save()

            with torch.no_grad():
                episode_to_buffer = self.gather_experiences()

            with torch.enable_grad():
                self.step_optimizer(episode_to_buffer)


    def get_episode_log_str(self):
        log_str = f'Ep {self.episode_counter}'
        log_str += f', PPO epochs: {self.ppo_epoch_counter}'
        log_str += f', net epochs: {self.network_epoch_counter}'
        log_str += f' | Nodes: {self.episodes_log["num_nodes"][-1]}'
        log_str += f' | Steps: {self.episodes_log["num_steps"][-1]}'
        # if isinstance(self.episodes_log['R'][-1], dict):
            # returns = {k: round(v, 3) for k, v in self.episodes_log['R'][-1].items()}
        # else:
            # returns = {self.agent_reward: round(self.episodes_log['R'][-1], 3)}
        #log_str += f' | Return(s): {round(self.episodes_log["R"][-1], 3)}'
        return log_str

    def get_epoch_log_str(self):
        log_str = f'Completed 1 PPO epoch in {round(self.epochs_log["ppo_epoch_run_time"][-1], 3)} s'
        log_str += f' | PPO epochs: {self.ppo_epoch_counter}'
        log_str += f' | Network epochs: {self.network_epoch_counter}'
        log_str += f' | Episodes: {self.episode_counter}'
        log_str += f' | g_loss: {np.mean(self.g_loss_episode)}'
        log_str += f' | e_loss: {np.mean(self.e_loss_episode)}'
        return log_str

    def calc_discounted_rewards(self, rewards, dones):
        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
        return discounted_rewards

    def whiten_rewards_array(self, rewards, eps=1e-4):
        rewards = np.array(rewards)
        if rewards.shape[0] > 1:
            return (rewards - rewards.mean()) / (rewards.std() + eps)
        else:
            return np.array([0], dtype=np.float32)

    def whiten_episode_to_rewards(self, episode_to_rewards):
        # extract rewards from each episode into a single flat array
        flattened_rewards = [r for rs in list(episode_to_rewards.values()) for r in rs]

        # whiten rewards across episodes
        whitened_rewards = self.whiten_rewards_array(flattened_rewards)

        # pack whitened rewards back into dict mapping episode to per-step rewards
        start_idx, episode_to_whitened_rewards = 0, {episode_id: [] for episode_id in episode_to_rewards.keys()}
        for episode_id in episode_to_rewards.keys():
            episode_length = len(episode_to_rewards[episode_id])
            episode_to_whitened_rewards[episode_id] = whitened_rewards[start_idx:start_idx+episode_length]
            start_idx += episode_length

        return episode_to_whitened_rewards

    def action_to_batch_idxs(self, action, state):
        '''Converts action from indexing action in each batch to indexing actions across all batches.'''
        return torch.cat([action[[0]], action[1:] + state.num_variables[:-1].cumsum(0)])

    def batch_buffer_experiences(self, _episode_to_buffer, epoch_stats=None):
        # unpack sub-trees (if present) and discount rewards in each episode
        episode_id, episode_to_discounted_rewards = 0, {}
        episode_to_buffer = {}
        for original_buffer in _episode_to_buffer.values():
            if not isinstance(original_buffer, list):
                # conv buffer to list so is iterable
                iterable_buffers = [original_buffer]
            else:
                # buffers already iterable
                iterable_buffers = original_buffer

            for _buffer in iterable_buffers:
                episode_to_buffer[episode_id] = _buffer
                episode_to_discounted_rewards[episode_id] = self.calc_discounted_rewards(_buffer.rewards, _buffer.dones)
                episode_id += 1
        if epoch_stats is not None:
            epoch_stats['discounted_returns'].append(np.mean([discounted_returns[0] for discounted_returns in episode_to_discounted_rewards.values()]))
        if self.debug_mode:
            print(f'episode_to_discounted_rewards: {episode_to_discounted_rewards}')

        if self.whiten_rewards:
            # whiten discounted rewards to reduce variance
            episode_to_discounted_rewards = self.whiten_episode_to_rewards(episode_to_discounted_rewards)
            if epoch_stats is not None:
                epoch_stats['whitened_discounted_returns'].append(np.mean([discounted_returns[0] for discounted_returns in episode_to_discounted_rewards.values()]))
            if self.debug_mode:
                print(f'episode_to_whitened_discounted_rewards: {episode_to_discounted_rewards}')

        # batch tensors
        batch_to_state = defaultdict(lambda: None)
        batch_to_action_idx = defaultdict(lambda: None)
        batch_to_logprob = defaultdict(lambda: None)
        batch_to_discounted_reward = defaultdict(lambda: None)

        batch_id, episode_id, step_idx = 0, 0, 0
        states, action_idxs, logprobs, discounted_rewards = [], [], [], []
        while True:
            if len(states) < self.batch_size:
                # stay on current batch
                pass
            else:
                # batch full, save
                batch_to_state[batch_id] = Batch.from_data_list(states).to(self.agent.device)
                batch_to_action_idx[batch_id] = torch.tensor(action_idxs, dtype=torch.int16).to(self.agent.device)
                batch_to_logprob[batch_id] = torch.tensor(logprobs, dtype=torch.float32).to(self.agent.device)
                batch_to_discounted_reward[batch_id] = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.agent.device)

                # start new batch
                states, action_idxs, logprobs, discounted_rewards = [], [], [], []
                batch_id += 1

            # add next experience to batch
            states.append(BipartiteNodeData(*episode_to_buffer[episode_id].states[step_idx]))
            action_idxs.append(episode_to_buffer[episode_id].action_idxs[step_idx])
            logprobs.append(episode_to_buffer[episode_id].logprobs[step_idx])
            discounted_rewards.append(episode_to_discounted_rewards[episode_id][step_idx])

            step_idx += 1
            if step_idx == len(episode_to_buffer[episode_id]):
                # ran out of experiences this episode, move to next episode
                episode_id += 1
                step_idx = 0

            if episode_id not in episode_to_buffer.keys() and len(states) != 0:
                # finished batching all experiences
                batch_to_state[batch_id] = Batch.from_data_list(states).to(self.agent.device)
                batch_to_action_idx[batch_id] = torch.tensor(action_idxs, dtype=torch.int16).to(self.agent.device)
                batch_to_logprob[batch_id] = torch.tensor(logprobs, dtype=torch.float32).to(self.agent.device)
                batch_to_discounted_reward[batch_id] = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.agent.device)
                break

        return batch_to_state, batch_to_action_idx, batch_to_logprob, batch_to_discounted_reward, epoch_stats

    def step_optimizer(self, episode_to_buffer):
        if self.debug_mode:
            print(f'\n>>> Stepping optimizer (epoch {self.ppo_epoch_counter+1} of {self.num_ppo_epochs}) <<<')

        # record epoch stats
        epoch_stats = defaultdict(lambda: [])

        # batch experiences in buffer
        batch_to_state, batch_to_action_idx, batch_to_logprob, batch_to_discounted_reward, epoch_stats = self.batch_buffer_experiences(episode_to_buffer, epoch_stats)

        #update discriminator para
        self.discriminator_optimizer.zero_grad()
        # shuffle batch_id order each PPO epoch to help with learning
        batch_ids = list(batch_to_state.keys())
        random.shuffle(batch_ids)

        max_batch = len(batch_ids)
        cur_batch_index = 0

        self.g_loss_episode = []
        self.e_loss_episode = []
        for batch in self.experience_loader:
            expert_batch = batch.to(self.agent.device)   

            if cur_batch_index >= max_batch:
                break
            batch_id =  batch_ids[cur_batch_index]    
            batched_state = batch_to_state[batch_id]
            batched_action_idx = batch_to_action_idx[batch_id]
            cur_batch_index +=1

            # calculate loss of discriminator
            g_o = self.agent.discriminator_network(batched_state.constraint_features, batched_state.edge_index, batched_state.edge_attr, batched_state.variable_features, candidates = batched_state.candidates, num_candidates = batched_state.num_candidates)
            g_o = g_o[0]
            result = g_o[torch.arange(g_o.size(0)), batched_action_idx.type(torch.long)]
            g_o = torch.sigmoid(result)
            

            e_o = self.agent.discriminator_network(expert_batch.constraint_features, expert_batch.edge_index, expert_batch.edge_attr, expert_batch.variable_features, candidates = expert_batch.candidates, num_candidates = expert_batch.num_candidates)
            e_o = e_o[0]
            result = e_o[torch.arange(e_o.size(0)), expert_batch.candidate_choices.type(torch.long)]
            e_o = torch.sigmoid(result)

            g_loss = self.discriminator_loss_function(g_o, torch.ones((g_o.shape[0]), device=self.agent.device))
            e_loss = self.discriminator_loss_function(e_o, torch.zeros((e_o.shape[0]), device=self.agent.device))
            discrim_loss = g_loss + e_loss
                
            discrim_loss.backward()
            self.discriminator_optimizer.step()
            
            self.g_loss_episode.append(g_loss.item())
            self.e_loss_episode.append(e_loss.item())

        if self.debug_mode:
            print(f'Optimizing PPO policy over {self.ppo_epochs_per_update} epochs...')

        start_t = time.time()
        for ppo_epoch in range(self.ppo_epochs_per_update):
            # optimize ppo policy
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            if self.debug_mode:
                print(f'> PPO epoch {ppo_epoch+1} of {self.ppo_epochs_per_update} <')

            # shuffle batch_id order each PPO epoch to help with learning
            batch_ids = list(batch_to_state.keys())
            random.shuffle(batch_ids)

            # accumulate gradients across batches
            for counter, batch_id in enumerate(batch_ids):
                # evaluate old actions and values with critic
                batched_state = batch_to_state[batch_id]
                batched_action_idx = batch_to_action_idx[batch_id]
                logprobs, state_values, dist_entropy = self.agent.evaluate_old_action(batched_action_idx, state=batched_state)
                if isinstance(state_values, list):
                    num_heads = len(state_values)
                else:
                    num_heads = 1
                if self.debug_mode:
                    print(f'-- batch_counter: {counter} --\nbatch_id: {batch_id}\nlogprobs: {logprobs}\nstate_values: {state_values}\ndist_entropy: {dist_entropy}')

                # calc ratios between current and old policy
                old_logprobs = batch_to_logprob[batch_id].detach()
                policy_ratios = torch.exp(logprobs - old_logprobs.detach())
                if epoch_stats is not None:
                    epoch_stats[f'policy_ratio'].append(torch.mean(policy_ratios).detach().cpu().tolist())
                    epoch_stats[f'logprob'].append(torch.mean(logprobs).detach().cpu().tolist())
                    epoch_stats[f'dist_entropy'].append(torch.mean(dist_entropy).detach().cpu().tolist())
                    for head in range(num_heads):
                        epoch_stats[f'head_{head}_state_value'].append(torch.mean(state_values[head]).detach().cpu().tolist())
                if self.debug_mode:
                    print(f'policy ratios: {policy_ratios}')

                # calc surrogate loss
                discounted_rewards = batch_to_discounted_reward[batch_id]
                discounted_rewards = [discounted_rewards]
                advantages = [discounted_rewards[head] - state_values[head].detach() for head in range(num_heads)]
                surr1 = [policy_ratios * advantages[head] for head in range(num_heads)]
                surr2 = [torch.clamp(policy_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[head] for head in range(num_heads)]
                if epoch_stats is not None:
                    for head in range(num_heads):
                        epoch_stats[f'head_{head}_advantage'].append(torch.mean(advantages[head]).detach().cpu().tolist())
                        epoch_stats[f'head_{head}_surr1'].append(torch.mean(surr1[head]).detach().cpu().tolist())
                        epoch_stats[f'head_{head}_surr2'].append(torch.mean(surr2[head]).detach().cpu().tolist())
                if self.debug_mode:
                    print(f'discounted_rewards: {discounted_rewards}')
                    print(f'advantages: {advantages}')
                    print(f'surr1: {surr1}')
                    print(f'surr2: {surr2}')

                # calc losses
                critic_loss = [self.value_function_coeff * self.critic_loss_function.extract(state_values[head], discounted_rewards[head], reduction='none') for head in range(num_heads)]
                actor_loss = [-torch.min(surr1[head], surr2[head]) - critic_loss[head] + (self.entropy_coeff * dist_entropy) for head in range(num_heads)]
                if self.debug_mode:
                    print(f'per-head actor loss: {actor_loss}')
                    print(f'per-head critic loss: {critic_loss}')

                # reduce losses across experiences
                if self.critic_loss_function.reduction == 'mean':
                    critic_loss = [torch.mean(critic_loss[head]) for head in range(num_heads)]
                    actor_loss = [torch.mean(actor_loss[head]) for head in range(num_heads)]
                else:
                    raise Exception(f'Unrecognised per-sample loss reduction {self.critic_loss_function.reduction}')
                if epoch_stats is not None:
                    for head in range(num_heads):
                        epoch_stats[f'head_{head}_actor_loss'].append(actor_loss[head].detach().cpu().tolist())
                        epoch_stats[f'head_{head}_critic_loss'].append(critic_loss[head].detach().cpu().tolist())
                if self.debug_mode:
                    print(f'per-head experience-reduced actor_loss: {actor_loss}')
                    print(f'per-head experience-reduced critic_loss: {critic_loss}')

                # sum losses across heads
                critic_loss = torch.sum(torch.stack(critic_loss), dim=-1)
                actor_loss = torch.sum(torch.stack(actor_loss), dim=-1)
                if self.debug_mode:
                    print(f'head-reduced actor_loss: {actor_loss}')
                    print(f'head-reduced critic_loss: {critic_loss}')

                # compute gradients
                actor_loss.mean().backward(retain_graph=True)
                critic_loss.mean().backward()

                if (counter + 1) % self.gradient_accumulation_factor == 0 or batch_id == batch_ids[-1]:

                    if self.actor_gradient_clipping_clip_value is not None:
                        torch.nn.utils.clip_grad_value_(self.agent.actor_network.parameters(), clip_value=self.actor_gradient_clipping_clip_value)
                    if self.critic_gradient_clipping_clip_value is not None:
                        torch.nn.utils.clip_grad_value_(self.agent.critic_network.parameters(), clip_value=self.critic_gradient_clipping_clip_value)

                    if self.path_to_save is not None:
                        # save actor gradients
                        actor_params = list(self.agent.actor_network.parameters())
                        actor_gradients = np.concatenate(np.array([actor_params[i].grad.detach().cpu().numpy().flatten() for i in range(len(actor_params))]))
                        epoch_stats['actor_mean_grad'].append(np.mean(actor_gradients))
                        epoch_stats['actor_min_grad'].append(np.min(actor_gradients))
                        epoch_stats['actor_max_grad'].append(np.max(actor_gradients))
                        epoch_stats['actor_std_grad'].append(np.std(actor_gradients))

                        # save critic gradients
                        critic_params = list(self.agent.critic_network.parameters())
                        critic_gradients = np.concatenate(np.array([critic_params[i].grad.detach().cpu().numpy().flatten() for i in range(len(critic_params))]))
                        epoch_stats['critic_mean_grad'].append(np.mean(critic_gradients))
                        epoch_stats['critic_min_grad'].append(np.min(critic_gradients))
                        epoch_stats['critic_max_grad'].append(np.max(critic_gradients))
                        epoch_stats['critic_std_grad'].append(np.std(critic_gradients))

                    # update networks
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                    self.network_epoch_counter += 1

                    # clear gradients
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    if self.debug_mode:
                        print(f'Updated network -> network_epoch_counter: {self.network_epoch_counter}')
        end_t = time.time()

        # copy updated network to old policy
        self.agent.old_actor_network.load_state_dict(self.agent.actor_network.state_dict())

        for key, val in epoch_stats.items():
            # record average of stat across ppo epochs
            epoch_stats[key] = np.mean(val)
        epoch_stats['num_network_epochs'] = self.network_epoch_counter
        epoch_stats['num_ppo_epochs'] = self.ppo_epoch_counter
        epoch_stats['num_episodes'] = self.episode_counter
        epoch_stats['ppo_epoch_run_time'] = end_t - start_t
        self.update_epochs_log(epoch_stats)

        if self.ppo_epoch_counter % self.epoch_log_freq == 0:
            print(self.get_epoch_log_str())

    def update_epochs_log(self, epoch_stats):
        for key, val in epoch_stats.items():
            self.epochs_log[key].append(val)

    def update_episodes_log(self, episode_stats):
        episode_stats['num_ppo_epochs'] = self.ppo_epoch_counter
        episode_stats['num_network_epochs'] = self.network_epoch_counter
        episode_stats['num_episodes'] = self.episode_counter
        episode_stats['elapsed_training_time'] = time.time() - self.train_start_t
        for key, val in episode_stats.items():
            self.episodes_log[key].append(val)

    def reset_episodes_log(self):
        # lists for logging stats at end of each episode
        self.episodes_log = defaultdict(list)

    def reset_epochs_log(self):
        self.epochs_log = defaultdict(list)

    def move_agent_to_device(self, agent, device):
        agent.to(device)
        return agent

















            



