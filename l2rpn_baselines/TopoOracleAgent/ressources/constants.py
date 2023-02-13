from grid2op.Rules import AlwaysLegal
from oracle4grid.core.agent.OracleOverloadReward import OracleOverloadReward
from oracle4grid.core.agent.OracleL2RPNReward import OracleL2RPNReward

# Backend class to use
try:
    from lightsim2grid.LightSimBackend import LightSimBackend
    BACKEND = LightSimBackend
except ModuleNotFoundError:
    from grid2op.Backend import PandaPowerBackend
    BACKEND = PandaPowerBackend


# Grid2Op Env constants
class EnvConstants:
    def __init__(self):
        # Reward class used for computing best path for oracle
        self.reward_class = OracleL2RPNReward
        # Other rewards to be computed
        self.other_rewards = {
            # Mandatory reward
            "overload_reward": OracleOverloadReward
            # Add other rewards here ->
        }
        # The game rule that the oracle will follow
        self.game_rule = AlwaysLegal
        # The G2OP params that the runner for the main simulation engine will use
        self.DICT_GAME_PARAMETERS_SIMULATION = {'NO_OVERFLOW_DISCONNECTION': True,
                                           'MAX_LINE_STATUS_CHANGED': 999,
                                           'MAX_SUB_CHANGED': 2999}
        # The G2OP params that the runner for the graph edges generation
        self.DICT_GAME_PARAMETERS_GRAPH = {'NO_OVERFLOW_DISCONNECTION': True,
                                      'MAX_LINE_STATUS_CHANGED': 1,
                                      'MAX_SUB_CHANGED': 1}
        # The G2OP params that the runner for the replay of the best found paths
        self.DICT_GAME_PARAMETERS_REPLAY = {'NO_OVERFLOW_DISCONNECTION': False,
                                       'MAX_LINE_STATUS_CHANGED': 1,
                                       'MAX_SUB_CHANGED': 1}

# Oracle constants
END_NODE_REWARD = 0.1

# Seed info
ENV_SEEDS = None
AGENT_SEEDS = None
