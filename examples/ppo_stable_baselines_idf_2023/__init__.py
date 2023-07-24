import os
import json
from grid2op.Agent import BaseAgent
from .C_evaluate_trained_model import load_agent, agent_name, gymenv_class


# two meta parameters not tune but that can be modified at inference time
safe_max_rho = 0.99
curtail_margin = 30.


# the required function for codanbench
def make_agent(env, this_directory_path):
    this_directory_path = os.path.abspath(this_directory_path)
    agent_path = os.path.join(this_directory_path, "saved_model_2023")
    with open(os.path.join(this_directory_path, "preprocess_obs.json"), "r", encoding="utf-8") as f:
        obs_space_kwargs = json.load(f)
    with open(os.path.join(this_directory_path, "preprocess_act.json"), "r", encoding="utf-8") as f:
        act_space_kwargs = json.load(f)
        
    my_agent = load_agent(env,
                          agent_path,
                          name=agent_name,
                          gymenv_class=gymenv_class,
                          gymenv_kwargs={"safe_max_rho": safe_max_rho},
                          obs_space_kwargs=obs_space_kwargs,
                          act_space_kwargs=act_space_kwargs)
    return BaselineAgent(my_agent, curtail_margin=curtail_margin)
