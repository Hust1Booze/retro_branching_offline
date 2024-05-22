"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from retro_branching.environments import EcoleBranching
from retro_branching.utils import BipartiteNodeData
from torch.nn import functional as F

logger = logging.getLogger(__name__)

#from utils import sample
from collections import deque
import random
import cv2
import torch
from PIL import Image
import torch_geometric 
import time
from collections import defaultdict, deque
import glob,ecole

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        path = self.config.ckpt_path + 'dt_'+ time.strftime('%Y-%m-%d-%H-%M-%S') + '.pt'
        logger.info("saving %s", path)
        torch.save(raw_model.state_dict(), path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            # loader = DataLoader(data, shuffle=True, pin_memory=True,
            #                     batch_size=config.batch_size,
            #                     num_workers=config.num_workers)
            loader = torch_geometric.data.DataLoader(data, batch_size=config.batch_size, shuffle=True) # defaule batch_size=32

            losses = []

            it =0
            for batch in loader:
                x,y,r,t = batch
                x = [x.constraint_features.to(self.device), x.edge_index.to(self.device), x.edge_attr.to(self.device), x.variable_features.to(self.device),
                     x.variable_features_nums.to(self.device), x.candidates_back.to(self.device), x.candidates_num.to(self.device)]
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    print(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                    it += 1

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        # best_loss = float('inf')
        
        best_return = -float('inf')

        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)
            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')

            # # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()

            if self.config.ckpt_path is not None:
                self.save_checkpoint()

            # -- pass in target returns
            if self.config.model_type == 'naive':
                eval_return = self.get_returns(0)
            elif self.config.model_type == 'reward_conditioned':
                if self.config.game == 'Breakout':
                    eval_return = self.get_returns(90)
                elif self.config.game == 'Seaquest':
                    eval_return = self.get_returns(1150)
                elif self.config.game == 'Qbert':
                    eval_return = self.get_returns(14000)
                elif self.config.game == 'Pong':
                    eval_return = self.get_returns(20)
                elif self.config.game == 'scip':
                    eval_return = self.get_returns_for_scip(-30)
                    
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    def get_returns(self, ret):
        self.model.train(False)
        args=Args(self.config.game.lower(), self.config.seed)
        env = Env(args)
        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        for i in range(10):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
        env.close()
        eval_return = sum(T_rewards)/10.
        print("target return: %d, eval return: %d" % (ret, eval_return))
        self.model.train(True)
        return eval_return

    def get_returns_for_scip(self, ret):

        # instances
        instances_path = f'/home/liutf/code/retro_branching_offline/retro_branching_paper_validation_instances/set_covering_n_rows_500_n_cols_1000'

        files = glob.glob(instances_path+f'/*.mps')
        instances = iter([ecole.scip.Model.from_file(f) for f in files])
        print(instances)
        print(f'Loaded {len(files)} instances from path {instances_path}')

        env = EcoleBranching(observation_function=self.config.observation_function,
                        information_function=self.config.information_function,
                        reward_function=self.config.reward_function,
                        scip_params=self.config.scip_params)
        env.seed(0)
        # metrics
        metrics = ['num_nodes', 'solving_time', 'lp_iterations']
        num_episod = 10

        validator = ValidatorForScip(agent=self.model.module,
                                            env=env,
                                            instances=instances,
                                            metrics=metrics,
                                            calibration_config_path=None,
#                                                calibration_config_path='/home/zciccwf/phd_project/projects/retro_branching/scripts/',
                                            seed=0,
                                            max_steps=1000000000000, # int(1e12), 10, 5, 3
                                            max_steps_agent=None,
                                            turn_off_heuristics=False,
                                            min_threshold_difficulty=None,
                                            max_threshold_difficulty=None, # None 250
                                            threshold_agent=None,
                                            threshold_env=None,
                                            episode_log_frequency=1,
                                            path_to_save=None,
                                            overwrite=None,
                                            checkpoint_frequency=10)
        steps = []
        start = time.time()
        for i in range(100):
            steps += [validator.run_episode(ret)]
        
        end = time.time()
        print(f'validator cost time :{end-start}')
        print(f'steps :{steps}')

    
class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


class ValidatorForScip():
    def __init__(self,
                 agent,
                 env,
                 instances,
                 calibration_config_path=None,
                 calibration_freq=10,
                 num_cpu_calibrations=15,
                 num_gpu_calibrations=500,
                 metrics=['num_nodes'],
                 seed=0,
                 max_steps=int(1e12),
                 max_steps_agent=None,
                 turn_off_heuristics=False,
                 max_threshold_difficulty=None,
                 min_threshold_difficulty=None,
                 episode_log_frequency=1,
                 checkpoint_frequency=1,
                 path_to_save=None,
                 overwrite=False,
                 name='rl_validator',
                 **kwargs):    
        self.agent = agent
        self.env = env
        self.instances = instances
        self.calibration_config_path = calibration_config_path
        self.calibration_freq = calibration_freq
        self.num_cpu_calibrations = num_cpu_calibrations
        self.num_gpu_calibrations = num_gpu_calibrations
        self.calibration_obs, self.calibration_action_set = None, None
        if self.calibration_config_path is not None:
            with open(self.calibration_config_path+'/cpu_calibration_config.json') as f:
                self.cpu_calibration_config = json.load(f)
            with open(self.calibration_config_path+'/gpu_calibration_config.json') as f:
                self.gpu_calibration_config = json.load(f)
            self.reset_calibation_time_arrays()
        self.metrics = metrics
        self.seed = seed
        self.max_steps = max_steps
        self.max_steps_agent = max_steps_agent
        self.turn_off_heuristics = turn_off_heuristics
        
        self.min_threshold_difficulty = min_threshold_difficulty
        self.max_threshold_difficulty = max_threshold_difficulty
        if self.min_threshold_difficulty is not None or self.max_threshold_difficulty is not None:
            # init threshold env for evaluating difficulty when generating instance
            if 'threshold_env' not in kwargs:
                self.threshold_env = EcoleBranching(observation_function=list(envs.values())[0].str_observation_function,
                                                   information_function=list(envs.values())[0].str_information_function,
                                                   reward_function=list(envs.values())[0].str_reward_function,
                                                   scip_params=list(envs.values())[0].str_scip_params)
                # self.threshold_env = EcoleBranching()
            else:
                self.threshold_env = kwargs['threshold_env']
            if 'threshold_agent' not in kwargs:
                self.threshold_agent = PseudocostBranchingAgent()
            else:
                self.threshold_agent = kwargs['threshold_agent']
            self.threshold_env.seed(self.seed)
            self.threshold_agent.eval() # put in evaluation mode
        else:
            self.threshold_env = None
            self.threshold_agent = None
        self.episode_log_frequency = episode_log_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_counter = 1
        self.path_to_save = path_to_save
        self.overwrite = overwrite
        self.name = name

        # init directory to save data
        if self.path_to_save is not None:
            self.path_to_save = self.init_save_dir(path=self.path_to_save)
        
        # ensure all envs have same seed for consistent resetting of instances
        env.seed(self.seed)
        
        self.curr_instance = None
        
        #self.episodes_log = self.init_episodes_log()

        self.kwargs = kwargs

        # try:
        #     self.device = self.agents[list(self.agents.keys())[0]].device
        # except AttributeError:
        #     # agent does not have device parameter, assume is e.g. strong branching and is on CPU
        #     self.device = 'cpu'
        self.device = 'cuda:0'

    def reset_env(self, env, max_attempts=10000):

        counter = 1
        while True:
            instance = next(self.instances)
            if self.turn_off_heuristics:
                instance = turn_off_scip_heuristics(instance)
            instance_before_reset = instance.copy_orig()
            env.seed(self.seed)
            obs, action_set, reward, done, info = env.reset(instance)
            # print(f'obs:{obs}\nreward: {reward}\ndone:{done}\ninfo:{info}')

            if not done:
                if self.min_threshold_difficulty is not None or self.max_threshold_difficulty is not None:
                    # check difficulty using threshold agent
                    meets_threshold = False
                    self.threshold_env.seed(self.seed)
                    _obs, _action_set, _reward, _done, _info = self.threshold_env.reset(instance_before_reset.copy_orig())
                    while not _done:
                        _action, _action_idx = self.threshold_agent.action_select(action_set=_action_set, obs=_obs, agent_idx=0)
                        # _action = _action_set[_action]
                        _obs, _action_set, _reward, _done, _info = self.threshold_env.step(_action)
                        if self.max_threshold_difficulty is not None:
                            if _info['num_nodes'] > self.max_threshold_difficulty:
                                # already exceeded threshold difficulty
                                break
                    if self.min_threshold_difficulty is not None:
                        if _info['num_nodes'] >= self.min_threshold_difficulty:
                            meets_threshold = True
                    if self.max_threshold_difficulty is not None:
                        if _info['num_nodes'] <= self.max_threshold_difficulty:
                            meets_threshold = True
                        else:
                            meets_threshold = False
                    if meets_threshold:
                        # can give instance to agent to learn on
                        self.curr_instance = instance_before_reset.copy_orig()
                        return obs, action_set, reward, done, info, instance_before_reset
                else:
                    self.curr_instance = instance_before_reset.copy_orig()
                    return obs, action_set, reward, done, info, instance_before_reset

            counter += 1
            if counter > max_attempts:
                raise Exception('Unable to generate valid instance after {} attempts.'.format(max_attempts))

    def init_episodes_log(self):
        episodes_log = {}
        for agent in self.agents.keys():
            episodes_log[agent] = defaultdict(list)
        episodes_log['metrics'] = self.metrics
        episodes_log['turn_off_heuristics'] = self.turn_off_heuristics
        episodes_log['min_threshold_difficulty'] = self.min_threshold_difficulty
        episodes_log['max_threshold_difficulty'] = self.max_threshold_difficulty
        episodes_log['agent_names'] = list(self.agents.keys())
                
        return episodes_log

    def extract_stats(self,obs,action_set):
        graph = BipartiteNodeData(obs.row_features, obs.edge_features.indices, 
                    obs.edge_features.values, obs.column_features,
                    candidates = action_set)
        return graph
    def run_episode(self,ret):
        max_steps = 300
        env = self.env
        try:
            _, _, _, _, _, instance_before_reset = self.reset_env(env=env)
        except StopIteration:
            # ran out of iteratons, cannot run any more episodes
            print('Ran out of iterations.')
            return
        all_states = []
        with torch.no_grad():
            episode_stats = defaultdict(list)
            env.seed(self.seed)
            obs, action_set, reward, done, info = env.reset(instance_before_reset.copy_orig())
            if action_set is not None:
                action_set = action_set.astype(int) # ensure action set is int so gets correctly converted to torch.LongTensor later
            
            state = self.extract_stats(obs,action_set)
            all_states += [state]
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = scip_sample(self.agent, all_states, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))  
            if sampled_action.item() not in action_set:
                print('error! action not in action_set')
            
            actions = []
            steps =0
            for t in range(self.max_steps):

                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                obs, action_set, reward, done, info = env.step(action)
                steps +=1
                if done or steps>max_steps:
                    break

                if action_set is not None:
                    action_set = action_set.astype(int) # ensure action set is int so gets correctly converted to torch.LongTensor later
                state = self.extract_stats(obs,action_set)
                all_states += [state]

                rtgs += [rtgs[-1] - (-1)]
                
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = scip_sample(self.agent, all_states, 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(steps, 10000) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
                if sampled_action.item() not in action_set:
                    print('error! action not in action_set')
            return steps
            #print(f'steps: {steps}')



class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4


##### add for dt

"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# import random
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if x.size(1) <= block_size//3 else x[:, -block_size//3:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:] # crop context if needed
        logits, _ = model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix

    return x

@torch.no_grad()
def scip_sample(model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if len(x) <= block_size//3 else x[-block_size//3:] # crop context if needed

        constraint_features = torch.cat([state.constraint_features for state in x_cond],dim =0)
        edge_index = torch.cat([state.edge_index for state in x_cond],dim =1)
        edge_attr = torch.cat([state.edge_attr for state in x_cond],dim =0)
        variable_features = torch.cat([state.variable_features for state in x_cond],dim =0)
        candidates = torch.cat([state.candidates for state in x_cond],dim =0)
        candidate_nums = torch.LongTensor([state.candidates.shape[0] for state in x_cond])
        variable_features_nums = torch.LongTensor([state.variable_features.shape[0] for state in x_cond])
        
        input = [constraint_features.to('cuda:0'),edge_index.to('cuda:0'),edge_attr.to('cuda:0'), variable_features.to('cuda:0'),\
                 variable_features_nums.to('cuda:0'), candidates.to('cuda:0'),candidate_nums.to('cuda:0')]
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:] # crop context if needed
        logits, _ = model(input, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix

    return x