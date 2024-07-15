import torch
import torch.nn.functional as F

from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads

import ml_collections
import itertools
import copy
import numpy as np
import json
import math


class GAILAgent:
    def __init__(self,
                 device=None,
                 actor_network=None,
                 critic_network=None,
                 discriminator_network=None,
                 head_aggregator='add',
                 num_heads=1,
                 config=None,
                 name='gail'):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if config is not None:
            self.init_from_config(config)
        else:
            self.actor_network = actor_network.to(self.device)
            self.critic_network = critic_network.to(self.device)

            self.old_actor_network = copy.deepcopy(actor_network).to(self.device)
            # self.old_critic_network = copy.deepcopy(critic_network)
            self.discriminator_network = discriminator_network.to(self.device)
            self.head_aggregator = head_aggregator
            self.name = name


    def init_from_config(self, config=None):
        if isinstance(config, str):
            # load from json
            with open(config, 'r') as f:
                json_config = json.load(f)
                config = ml_collections.ConfigDict(json.loads(json_config))

        self.actor_network = BipartiteGCN(device=self.device,
                                           config=config.actor_network)
        self.actor_network.to(self.device)
        self.critic_network = BipartiteGCN(device=self.device,
                                           config=config.critic_network)
        self.critic_network.to(self.device)

        self.old_actor_network = copy.deepcopy(self.actor_network)
        # self.old_critic_network = copy.deepcopy(self.critic_network)

        self.discriminator_network = BipartiteGCN(device=self.device,
                                           config=config.discriminator_network)
        self.discriminator_network.to(self.device)

        for key, val in config.agent.items():
            if key != 'device':
                self.__dict__[key] = val

    def get_networks(self):
        return {'actor_network': self.actor_network,
                'critic_network': self.critic_network,
                'old_actor_network': self.old_actor_network,
                'discriminator_network': self.discriminator_network,
                }
        

    def before_reset(self, model):
        pass

    def create_config(self):
        '''Returns config dict so that can re-initialise easily.'''
        # create agent dict of self.<attribute> key-value pairs
        agent_dict = {}
        for key, val in self.__dict__.items():
            # remove NET() networks to avoid circular references and no need to save torch tensors
            if type(val) != torch.Tensor and key not in list(self.get_networks().keys()):
                agent_dict[key] = val

        # create config dict
        config = {'agent': ml_collections.ConfigDict(agent_dict),
                  'actor_network': self.actor_network.create_config(),
                  'critic_network': self.critic_network.create_config(),
                  'discriminator_network': self.critic_network.create_config(),
                  }

        config = ml_collections.ConfigDict(config)

        return config

    def _parse_state(self, **kwargs):
        if 'state' not in kwargs:
            if 'action_set' not in kwargs and 'obs' not in kwargs:
                raise Exception('Must provide either state or action_set and obs as kwargs.')

        if 'state' in kwargs:
            self.obs = (kwargs['state'].constraint_features, kwargs['state'].edge_index, kwargs['state'].edge_attr, kwargs['state'].variable_features)
            self.action_set = torch.as_tensor(kwargs['state'].candidates)
        else:
            # unpack
            self.action_set, self.obs = kwargs['action_set'], kwargs['obs']
            if isinstance(self.action_set, np.ndarray):
                self.action_set = torch.as_tensor(self.action_set)
        return self.action_set, self.obs, kwargs

    def get_logits(self, net, obs):
        if isinstance(obs, tuple):
            logits = net(*obs)
        else:
            logits = net(obs)
        return logits

    def process_heads(self, logits):
        if isinstance(logits, list):
            # multiple heads, need to aggregate to get values for each action
            _logits = [logits[head] for head in range(len(logits))]

            # get head aggregator
            if isinstance(self.head_aggregator, dict):
                if self.training:
                    head_aggregator = self.head_aggregator['train']
                else:
                    head_aggregator = self.head_aggregator['test']
            else:
                head_aggregator = self.head_aggregator

            if head_aggregator is None:
                _logits = torch.stack(_logits).squeeze(0)
            elif head_aggregator == 'add':
                _logits = torch.stack(_logits, dim=0).sum(dim=0)
            elif isinstance(head_aggregator, int):
                _logits = _logits[head_aggregator]
            else:
                raise Exception(f'Unrecognised head_aggregator {self.head_aggregator}')
        else:
            # no heads
            pass
        return _logits

    def action_select(self, **kwargs):
        # use old actor network
        self.action, self.action_set, self.obs, self.dist, self.action_logprob, kwargs = self.network_action_select(self.old_actor_network, **kwargs)

        return self.action, self.action_idx

    def network_action_select(self, net, **kwargs):
        self.action_set, self.obs, kwargs = self._parse_state(**kwargs)

        # get masked network output
        self.logits = self.get_logits(net, self.obs) # forward pass through network
        self.logits = self.process_heads(self.logits) # handle any multi-head outputs
        self.logits = self.logits[self.action_set] # mask invalid action logits

        if 'state' in kwargs:
            # batch of observations
            self.logits = self.logits.split_with_sizes(tuple(kwargs['state'].num_candidates))
            self.action_set = kwargs['state'].raw_candidates.split_with_sizes(tuple(kwargs['state'].num_candidates))
            if not self.training:
                # deterministically select greedy action
                self.action_idx = torch.stack([logits.argmax() for logits in self.logits])
            else:
                # stochastically select action
                # self.preds = [F.softmax(preds/kwargs['munchausen_tau'], dim=0) for preds in self.preds]
                self.probs = [F.softmax(logits, dim=0) for logits in self.logits]
                self.dist = [torch.distributions.Categorical(probs) for probs in self.probs] # init discrete categorical distribution from softmax probs
                self.action_idx = [dist.sample() for dist in self.dist] # sample action from categorical distribution
                self.action_logprob = torch.stack([dist.log_prob(action_idx) for dist, action_idx in zip(self.dist, self.action_idx)])
            self.action = torch.stack([_action_set[idx] for _action_set, idx in zip(self.action_set, self.action_idx)])
            self.action_idx = torch.stack(self.action_idx) # conv list to tensor
        else:
            # single observation
            if not self.training:
                # deterministically select greedy action
                self.action_idx = self.logits.argmax()
            else:
                # stochastically select action
                self.probs = F.softmax(self.logits, dim=0)
                self.dist = torch.distributions.Categorical(self.probs)
                self.action_idx = self.dist.sample()
                self.action_logprob = self.dist.log_prob(self.action_idx)
            self.action = self.action_set[self.action_idx.item()]

        return self.action, self.action_set, self.obs, self.dist, self.action_logprob, kwargs

    def evaluate_old_action(self, action_idx, **kwargs):
        '''
        action_idx must be the idx in action_set of the original action so that
        can re-index the categorical distribution.
        '''
        # use current actor network
        _, self.action_set, self.obs, self.dist, _, kwargs = self.network_action_select(self.actor_network, **kwargs)
        if isinstance(self.dist, list):
            self.action_logprob = torch.stack([self.dist[i].log_prob(action_idx[i]) for i in range(len(self.dist))])
            self.dist_entropy = torch.stack([dist.entropy() for dist in self.dist])
        else:
            self.action_logprob = self.dist.log_prob(action_idx)
            self.dist_entropy = self.dist.entropy()

        # DEBUG
        # print(f'eval action_set: {self.action_set}')
        # print(f'eval action_idx: {action_idx}')

        # NEW: Just use critic's max output as value
        self.logits = self.get_logits(self.critic_network, self.obs)
        if isinstance(self.action_set, tuple):
            logits = [self.logits[head].split_with_sizes(tuple(kwargs['state'].num_variables)) for head in range(len(self.logits))]
            self.state_value = [torch.stack([torch.max(logits[head][i]) for i in range(action_idx.shape[0])]) for head in range(len(logits))]
        else:
            self.state_value = [torch.max(self.logits[head]) for head in range(len(self.logits))]

        return self.action_logprob, self.state_value, self.dist_entropy

    def train(self):
        self.actor_network.train()
        self.critic_network.train()
        self.old_actor_network.train()
        self.discriminator_network.train()
        self.training = True

    def eval(self):
        self.actor_network.eval()
        self.critic_network.eval()
        self.old_actor_network.eval()
        self.discriminator_network.eval()
        self.training = False

    # this function use to get reward from discriminator_network for PPO 
    def get_expert_reward(self, state, action_set, action_idx):
        with torch.no_grad():
            logits = self.discriminator_network(state, candidates = action_set, num_candidates = len(action_set))
            # first 0 for first head output , second 0 for first of n, in this function only input one state
            prob =  torch.sigmoid(logits[0][0][action_idx.item()]) 
            return -math.log(prob)



















