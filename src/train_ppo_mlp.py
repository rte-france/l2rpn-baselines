import grid2op
from l2rpn_baselines.PPO_SB3 import train, evaluate
import re
from grid2op.Reward import LinesCapacityReward
from grid2op.Chronics import MultifolderWithCache
from lightsim2grid import LightSimBackend

env_name = "l2rpn_case14_sandbox"
env = grid2op.make(
    env_name + "_train",
    reward_class=LinesCapacityReward,
    backend=LightSimBackend(),
    chronics_class=MultifolderWithCache,
)
val_env = grid2op.make(
    env_name + "_val",
    reward_class=LinesCapacityReward,
    backend=LightSimBackend(),
    chronics_class=MultifolderWithCache,
)

env.chronics_handler.real_data.set_filter(lambda x: re.match(".*", x) is not None)
env.chronics_handler.real_data.reset()
val_env.chronics_handler.real_data.set_filter(lambda x: re.match(".*", x) is not None)
val_env.chronics_handler.real_data.reset()
res = train(
    env,
    save_path="runs",
    iterations=1000000,
    logs_dir="runs/logs",
    save_every_xxx_steps=10000,
    eval_every_xxx_steps=1000,
    eval_env=val_env,
)
res = evaluate(val_env, load_path=".", nb_episode=10, verbose=True)
