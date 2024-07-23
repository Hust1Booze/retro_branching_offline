import ecole
import numpy as np
from collections import defaultdict

class Focus_Node():
    '''


    '''
    def __init__(self, *args, **kwargs):
        pass
    def before_reset(self, model):
        pass
    def extract(self, model, done):

        m = model.as_pyscipopt()
        # get current node (i.e. next node to be branched at)
        cur_node = None
        parent_node = None
        cur_node_id = None
        parent_node_id = None

        cur_node = m.getCurrentNode()
        if cur_node is not None:
            cur_node_id = cur_node.getNumber()

            parent_node = cur_node.getParent()
            if parent_node is not None:
                parent_node_id = parent_node.getNumber()
                
        return [cur_node_id,parent_node_id]