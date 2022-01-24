import gym
import json
import os
import grid2op
import re
from grid2op.Reward import L2RPNReward, EpisodeDurationReward
from grid2op.gym_compat import GymEnv, DiscreteActSpace, BoxGymObsSpace
from l2rpn_baselines.DuelQSimple import train
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache
from stable_baselines3 import PPO

# define the environment
env = grid2op.make("l2rpn_case14_sandbox",
                    reward_class=EpisodeDurationReward,
                    backend=LightSimBackend(),
                    chronics_class=MultifolderWithCache)

env.chronics_handler.real_data.set_filter(lambda x: re.match(".*00$", x) is not None)
env.chronics_handler.real_data.reset()
env_gym = GymEnv(env)

li_act_path = "line_act.json"
if os.path.exists(li_act_path):
    with open(li_act_path, "r", encoding="utf-8") as f:
        all_acts = json.load(f)
else:
    all_acts = [env.action_space().as_serializable_dict()]
    for el in range(env.n_line):
        all_acts.append(env.action_space({"set_line_status" : [(el, -1)]}).as_serializable_dict())
        all_acts.append(env.action_space({"set_line_status" : [(el, +1)]}).as_serializable_dict())

env_gym.action_space = DiscreteActSpace(env.action_space, action_list= all_acts)

li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                 "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                 "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]
env_gym.observation_space =  BoxGymObsSpace(env.observation_space, attr_to_keep=li_attr_obs_X)

# learn
model = PPO("MlpPolicy", env_gym, verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=100_000)


# test
obs = env_gym.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env_gym.step(action)
    if done:
        print(f"{reward=}")
        obs = env_gym.reset()

env.close()