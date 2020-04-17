from grid2op.Agent import BaseAgent

class DoNothing(BaseAgent):
    """
    Do nothing agent of grid2op, as a lowerbond baseline for l2rpn competition.
    """
    def __init__(self,
                 action_space,
                 observation_space,
                 name,
                 **kwargs):
        super().__init__(action_space)
        self.name = name

    def act(self, observation, reward, done):
        # Just return an empty action aka. "do nothing"
        return self.action_space({})

    def reset(self, observation):
        # No internal states to reset
        pass

    def load(self, path):
        # Nothing to load
        pass

    def save(self, path):
        # Nothing to save
        pass

