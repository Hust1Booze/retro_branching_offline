import ecole
import numpy as np
from collections import defaultdict

class NodeBipariteWithIdx(ecole.observation.NodeBipartite):
    '''
    pyscipopt doesnt provide get brother node method, so here record curr and parent node idx,
    after instance solved, obtain parent and child nodes through traversal
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def before_reset(self, model):
        super().before_reset(model)
        
        self.init_dual_bound = None
        self.init_primal_bound = None
        
        
    def extract(self, model, done):
        # get the NodeBipartite obs
        obs = super().extract(model, done)
        
        m = model.as_pyscipopt()
        curr_node = m.getCurrentNode()
        curr_idx = curr_node.getNumber()

        parent_node = curr_node.getParent()
        parent_idx = -1
        if parent_node is not None:
            parent_idx = parent_node.getNumber()

        return obs, curr_idx, parent_idx