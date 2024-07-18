import copy

class DualBoundChange:
    def __init__(self, sense=-1): 
        '''
        Args:
            sense (-1, 1): -1 to minimise, +1 to maximise.
        '''
        self.sense = sense

    def before_reset(self, model):
        # m = model.as_pyscipopt()
        # self.internal_dual_bound = m.getDualbound()
        self.init_dual_bound = None

    def extract(self, model, done):
        '''Updates the internal dual bound and returns the difference multiplied by the sense.'''
        m = model.as_pyscipopt()
        if self.init_dual_bound is None:
            self.init_dual_bound = m.getDualbound()
            self.dual_bound = m.getDualbound()
            return 0
        self.prev_dual_bound = copy.deepcopy(self.dual_bound)
        self.dual_bound = m.getDualbound()
        reward = self.sense*(self.prev_dual_bound-self.dual_bound)
        return reward 