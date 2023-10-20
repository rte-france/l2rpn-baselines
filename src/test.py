import grid2op
from l2rpn_baselines.PPO_SB3 import evaluate
from l2rpn_baselines.DoNothing import evaluate as evaluate_do_nothing
import re
from grid2op.Reward import LinesCapacityReward
from grid2op.Chronics import MultifolderWithCache
from lightsim2grid import LightSimBackend

env_name = "l2rpn_case14_sandbox"
env_name = env_name + "_test"
env = grid2op.make(env_name,
                        reward_class=LinesCapacityReward,
                        backend=LightSimBackend(),
                        chronics_class=MultifolderWithCache)
env.chronics_handler.real_data.set_filter(lambda x: re.match(".*", x) is not None)
env.chronics_handler.reset()
res_agent = evaluate(env, load_path="runs", nb_episode=8, verbose=True)
res = evaluate_do_nothing(env, nb_episode=8, verbose=True)
