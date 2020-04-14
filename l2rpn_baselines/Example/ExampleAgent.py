from grid2op.Agent import DoNothingAgent


class ExampleAgent(DoNothingAgent):
    """
    Do nothing agent of grid2op, as a lowerbond baseline for l2rpn competition
    """
    def __init__(self,
                 action_space,
                 observation_space):
        DoNothingAgent.__init__(self, action_space)

    def load(self, path):
        pass

    def save(self, path):
        pass

    def train(self, num_pre_training_steps, num_training_steps):
        pass