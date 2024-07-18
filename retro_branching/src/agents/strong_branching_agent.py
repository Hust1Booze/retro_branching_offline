import ecole
import numpy as np
from datetime import datetime
import os
import pyscipopt

class StrongBranchingAgent:
    def __init__(self, pseudo_candidates=False, name='sb'):
        self.name = name
        self.pseudo_candidates = pseudo_candidates
        self.strong_branching_function = ecole.observation.StrongBranchingScores(pseudo_candidates=pseudo_candidates)

        self.BBTree = BBTree(top_k=30,use_root_scores = True)
        self.use_root_sb = False

    def before_reset(self, model):
        """
        This function will be called at initialization of the environments (before dynamics are reset).
        """
        self.strong_branching_function.before_reset(model)
    
    def extract(self, model, done): 
        self.BBTree.start_new = False
        return self.strong_branching_function.extract(model, done)

    def action_select(self, action_set, model, done, **kwargs):
        m = model.as_pyscipopt()
        # get current node (i.e. next node to be branched at)
        self._curr_node = None
        self._parent_node = None
        self.curr_node_id = None
        self.parent_node_id = None
        _curr_node = m.getCurrentNode()
        solvingtime = 0
        if _curr_node is not None:
            self.curr_node_id = _curr_node.getNumber()

            self._parent_node = _curr_node.getParent()
            if self._parent_node is not None:
                self.parent_node_id = self._parent_node.getNumber()

        scores = None
        if self.BBTree.start_new:
            scores = self.extract(model, done)
        elif self.use_root_sb is False:
            scores = self.extract(model, done)

        sorted_index = np.argsort(scores[action_set])
        #print(action_set[top_k_indices])
        action_idx = scores[action_set].argmax()
        action = action_set[action_idx]

        self.BBTree.add_node(self.curr_node_id, self.parent_node_id, action_set[action_idx], scores[action_set[action_idx]],action_set[sorted_index], scores[action_set[sorted_index]])

        return action, action_idx 

        

        
        if  action_set[action_idx] != action_set[sorted_index[-1]]:
            assert scores[action_set[action_idx]] == scores[action_set[sorted_index[-1]]]

        
        if self.BBTree.use_root_scores:
            action = -1 
            while action not in action_set:
                action = self.BBTree.get_root_scores_action()
            self.BBTree.update_last_log(action, scores[action])
        
        return action, action_idx
    
    def action_select_sb(self, action_set, model, done, **kwargs):

        scores = self.extract(model, done)
        if self.BBTree.start_new:              
            sorted_index = np.argsort(scores[action_set])

            self.BBTree.add_node(1, None, None, None, action_set[sorted_index], scores[action_set[sorted_index]])
            self.BBTree.start_new = False

        else:
            scores = []
            self.BBTree.add_node(None, None, None, None, None, None)


        action = -1 
        while action not in action_set:
            action = self.BBTree.get_root_scores_action()
        return action, None
        #self.BBTree.update_last_log(action, scores[action])
    

class BBTree():
    def __init__(self,top_k = 30,use_root_scores = False):
        self.top_k = top_k
        self.node_id = []
        self.parenet_node_id = []
        self.actions = []
        self.scores = []
        self.sorted_action = []
        self.sorted_scores = []

        self.root_scores = []
        self.use_root_scores = use_root_scores

        self.start_new = False
    def add_node(self, node_id, parenet_node_id, action, action_score, sorted_action, sorted_score):
        if node_id == 1:
            self.root_scores =  sorted_score
            self.root_actions = sorted_action
            self.used_action = [] # avoid one action use twice, dont know if right?

        self.node_id.append(node_id)
        self.parenet_node_id.append(parenet_node_id)
        self.actions.append(action)
        self.scores.append(action_score)
        self.sorted_action.append(sorted_action)
        self.sorted_scores.append(sorted_score)

    def get_root_scores_action(self):
        for action in reversed(self.root_actions):
            if action not in self.used_action:
                self.used_action.append(action)
                return action
        # if all action have been used
        self.used_action = []
        return self.get_root_scores_action()
    
    # after use root_scores_action, need update last action and its score
    def update_last_log(self, action, score):
        self.actions[-1] = action
        self.scores[-1] = score
    
    def show_tree(self):
        for i in range(len(self.node_id)):
            if self.node_id[i] == 1:
                print('*'*50)
            print(f'node: {self.node_id[i]}\n,  \
                  parent_node: {self.parenet_node_id[i]}\n, \
                  action: {self.actions[i]}\n, \
                  score: {self.scores[i]}\n, \
                  sorted_actions: {self.sorted_action[i]}\n, \
                  sorted_scores: {self.sorted_scores[i]}\n')
    
    def save_log(self):
        now = datetime.now()  # 获得当前时间
        global_timestr = now.strftime("%m_%d_%H_%M")
        self.log_dir = f'SB_'
        self.log_file = self.log_dir + 'log.txt'
        f = open(self.log_file, 'a')

        for i in range(len(self.node_id)):
            if self.node_id[i] == 1:
                f.write('*'*50 + '\n')
            f.write(f'node: {self.node_id[i]}\n parent_node: {self.parenet_node_id[i]}\n action: {self.actions[i]}\n score: {self.scores[i]}\n sorted_action: {self.sorted_action[i][-self.top_k:]}\n sorted_score: {self.sorted_scores[i][-self.top_k:]}\n')

        f.close()


    

