import pdb
import grid2op
import re
import os
from grid2op.Reward import L2RPNReward
from grid2op.Runner import Runner
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache

# define the environment
env = grid2op.make("l2rpn_case14_sandbox",
                    reward_class=L2RPNReward,
                    backend=LightSimBackend(),
                    # chronics_class=MultifolderWithCache
                    )

# env.chronics_handler.real_data.set_filter(lambda x: re.match(".*00$", x) is not None)
# env.chronics_handler.real_data.reset()
runner = Runner(**env.get_params_for_runner())
res = runner.run(nb_episode=10, episode_id=["0000", "0100", "0200", "0300", "0400", "0500", "0600", "0700", "0800", "0900"])
pdb.set_trace()
{'0000': 1091, '0100': 1097, '0300': 1096,  '0400': 2828, '0500': 514, '0600': 1091, '0700': 717, '0800': 513, '0900': 381}
# mean time survived: 1036.4444444444443