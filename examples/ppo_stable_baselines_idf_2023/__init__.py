import os
import json
from grid2op.Agent import BaseAgent
from .C_evaluate_trained_model import load_agent, agent_name
from .gymenv_custom import GymEnvWithRecoWithDN as gymenv_class

safe_max_rho = 0.99
curtail_margin = 30.

class BaselineAgent(BaseAgent):
  def __init__(self, l2rpn_agent, curtail_margin=150.):
    self.l2rpn_agent = l2rpn_agent
    self.curtail_margin = curtail_margin
    BaseAgent.__init__(self, l2rpn_agent.action_space)
  
  def act(self, obs, reward, done=False):
    action = self.l2rpn_agent.act(obs, reward, done)
    # We try to limit to end up with a "game over" because actions on curtailment or storage units.
    action.limit_curtail_storage(obs, margin=self.curtail_margin)
    return action

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
