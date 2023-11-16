import grid2op
from grid2op.Reward import LinesCapacityReward
from grid2op.Chronics import MultifolderWithCache
from lightsim2grid import LightSimBackend

env_name = "l2rpn_case14_sandbox"
env = grid2op.make(
    env_name,
    reward_class=LinesCapacityReward,
    backend=LightSimBackend(),
    chronics_class=MultifolderWithCache,
)

nm_env_train, nm_env_val, nm_env_test = env.train_val_split_random(
    add_for_test="test", pct_val=20.0, pct_test=10.0
)


# and now you can use the training set only to train your agent:
print(f"The name of the training environment is \\{nm_env_train}\\")
print(f"The name of the validation environment is \\{nm_env_val}\\")
print(f"The name of the test environment is \\{nm_env_test}\\")
