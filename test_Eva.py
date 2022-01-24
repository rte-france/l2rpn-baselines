import json
import os
import grid2op
import re
from grid2op.Reward import L2RPNReward, EpisodeDurationReward
from l2rpn_baselines.utils import TrainingParam, NNParam
from l2rpn_baselines.DuelQSimple import train
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache

# define the environment
env = grid2op.make("l2rpn_case14_sandbox",
                    reward_class=EpisodeDurationReward,
                    backend=LightSimBackend(),
                    chronics_class=MultifolderWithCache)

env.chronics_handler.real_data.set_filter(lambda x: re.match(".*00$", x) is not None)
env.chronics_handler.real_data.reset()

# use the default training parameters
tp = TrainingParam()

# this will be the list of what part of the observation I want to keep
# more information on https://grid2op.readthedocs.io/en/latest/observation.html#main-observation-attributes
li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                 "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                 "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]

# neural network architecture
observation_size = NNParam.get_obs_size(env, li_attr_obs_X)
sizes = [800, 494, 494]  # sizes of each hidden layers
kwargs_archi = {'observation_size': observation_size,
                'sizes': sizes,
                'activs': ["relu" for _ in sizes],  # all relu activation function
                "list_attr_obs": li_attr_obs_X}
li_act_path = "line_act.json"
if os.path.exists(li_act_path):
    with open(li_act_path, "r", encoding="utf-8") as f:
        all_acts = json.load(f)
else:
    all_acts = [env.action_space().as_serializable_dict()]
    for el in range(env.n_line):
        all_acts.append(env.action_space({"set_line_status" : [(el, -1)]}).as_serializable_dict())
        all_acts.append(env.action_space({"set_line_status" : [(el, +1)]}).as_serializable_dict())
    
    with open(li_act_path, "w", encoding="utf-8") as f:
       json.dump(fp=f, obj=all_acts)

# select some part of the action
# more information at https://grid2op.readthedocs.io/en/latest/converter.html#grid2op.Converter.IdToAct.init_converter
kwargs_converters = {"all_actions": all_acts }
# define the name of the model
nm_ = "AnneOnymous6"
try:
    train(env,
          name=nm_,
          iterations=1_000_000,
          save_path="./saved_agents",
          load_path=None,
          logs_dir="./logs",
          training_param=tp,
          kwargs_converters=kwargs_converters,
          kwargs_archi=kwargs_archi)
finally:
    env.close()
