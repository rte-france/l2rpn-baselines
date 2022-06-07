import shutil
import numpy as np
from l2rpn_baselines.DuelQSimple import train
from l2rpn_baselines.utils import NNParam, TrainingParam
from grid2op import make

def filter_action_fun(grid2op_act):
    # filter out all non redispatching actions
    if np.any(grid2op_act.set_bus != 0):
        return False
    if np.any(grid2op_act.change_bus):
        return False
    if np.any(grid2op_act.curtail != -1.):
        return False
    if np.any(grid2op_act.storage_p != 0):
        return False
    if np.any(grid2op_act.line_set_status != 0):
        return False
    if np.any(grid2op_act.line_change_status):
        return False
    # it should be a redispatching action
    return True

if __name__ == "__main__":

    train_iter = 1000
    env_name = "l2rpn_case14_sandbox"

    env = make(env_name)  


    agent_name = "test_agent"
    save_path = "saved_agent_DDDQN_{}".format(train_iter)
    shutil.rmtree(save_path, ignore_errors=True)
    logs_dir="tf_logs_DDDQN"

    li_attr_obs_X = ["gen_p", "gen_v", "load_p", "load_q"]

    observation_size = NNParam.get_obs_size(env, li_attr_obs_X) 

    sizes = [300, 300, 300]  # 3 hidden layers, of 300 units each, why not...
    activs =  ["relu" for _ in sizes]  # all followed by relu activation, because... why not

    kwargs_archi = {'observation_size': observation_size,
                    'sizes': sizes,
                    'activs': activs,
                    "list_attr_obs": li_attr_obs_X}

    # baselines.readthedocs.io/en/latest/utils.html#l2rpn_baselines.utils.TrainingParam
    tp = TrainingParam()
    tp.batch_size = 32  # for example...
    tp.update_tensorboard_freq = int(train_iter / 10)
    tp.save_model_each = int(train_iter / 3)
    tp.min_observation = int(train_iter / 5)
    train(env,
        name=agent_name,
        iterations=train_iter,
        save_path=save_path,
        load_path=None, # put something else if you want to reload an agent instead of creating a new one
        logs_dir=logs_dir,
        kwargs_archi=kwargs_archi,
        training_param=tp,
        filter_action_fun=filter_action_fun)

