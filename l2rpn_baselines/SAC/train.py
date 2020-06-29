#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import tensorflow as tf
import warnings

from l2rpn_baselines.utils import cli_train
from l2rpn_baselines.SAC.SAC import SAC, DEFAULT_NAME
from l2rpn_baselines.SAC.SAC_NNParam import SAC_NNParam
from l2rpn_baselines.SAC.SAC_NN import SAC_NN
from l2rpn_baselines.utils import TrainingParam
from l2rpn_baselines.utils.waring_msgs import _WARN_GPU_MEMORY


def train(env,
          name=DEFAULT_NAME,
          iterations=1,
          save_path=None,
          load_path=None,
          logs_dir=None,
          training_param=None,
          filter_action_fun=None,
          verbose=True,
          kwargs_converters={},
          kwargs_archi={}):
    """
    This function implements the "training" part of the balines "DeepQSimple".

    Parameters
    ----------
    env: :class:`grid2op.Environment`
        Then environment on which you need to train your agent.

    name: ``str```
        The name of your agent.

    iterations: ``int``
        For how many iterations (steps) do you want to train your agent. NB these are not episode, these are steps.

    save_path: ``str``
        Where do you want to save your baseline.

    load_path: ``str``
        If you want to reload your baseline, specify the path where it is located. **NB** if a baseline is reloaded
        some of the argument provided to this function will not be used.

    logs_dir: ``str``
        Where to store the tensorboard generated logs during the training. ``None`` if you don't want to log them.

    verbose: ``bool``
        If you want something to be printed on the terminal (a better logging strategy will be put at some point)

    training_param: :class:`l2rpn_baselines.utils.TrainingParam`
        The parameters describing the way you will train your model.

    filter_action_fun: ``function``
        A function to filter the action space. See
        `IdToAct.filter_action <https://grid2op.readthedocs.io/en/latest/converter.html#grid2op.Converter.IdToAct.filter_action>`_
        documentation.

    kwargs_converters: ``dict``
        A dictionary containing the key-word arguments pass at this initialization of the
        :class:`grid2op.Converter.IdToAct` that serves as "Base" for the Agent.

    kwargs_archi: ``dict``
        Key word arguments used for making the :class:`DeepQ_NNParam` object that will be used to build the baseline.

    Returns
    -------

    baseline: :class:`DeepQSimple`
        The trained baseline.


    .. _Example-sac:

    Examples
    ---------
    Here is an example on how to train a SAC baseline.

    First define a python script, for example

    .. code-block:: python

        import grid2op
        from grid2op.Reward import L2RPNReward
        from l2rpn_baselines.utils import TrainingParam, NNParam
        from l2rpn_baselines.SAC import train

        # define the environment
        env = grid2op.make("l2rpn_case14_sandbox",
                           reward_class=L2RPNReward)

        # use the default training parameters
        tp = TrainingParam()

        # this will be the list of what part of the observation I want to keep
        # more information on https://grid2op.readthedocs.io/en/latest/observation.html#main-observation-attributes
        li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                         "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                         "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]

        # neural network architecture
        observation_size = NNParam.get_obs_size(env, li_attr_obs_X)
        sizes_q = [800, 800, 800, 494, 494, 494]  # sizes of each hidden layers
        sizes_v = [800, 800]  # sizes of each hidden layers
        sizes_pol = [800, 800, 800, 494, 494, 494]  # sizes of each hidden layers
        kwargs_archi = {'observation_size': observation_size,
                        'sizes': sizes_q,
                        'activs': ["relu" for _ in range(len(sizes_q))],
                        "list_attr_obs": li_attr_obs_X,
                        "sizes_value": sizes_v,
                        "activs_value": ["relu" for _ in range(len(sizes_v))],
                        "sizes_policy": sizes_pol,
                        "activs_policy": ["relu" for _ in range(len(sizes_pol))]
                        }

        # select some part of the action
        # more information at https://grid2op.readthedocs.io/en/latest/converter.html#grid2op.Converter.IdToAct.init_converter
        kwargs_converters = {"all_actions": None,
                             "set_line_status": False,
                             "change_bus_vect": True,
                             "set_topo_vect": False
                             }
        # define the name of the model
        nm_ = "AnneOnymous"
        try:
            train(env,
                  name=nm_,
                  iterations=10000,
                  save_path="/WHERE/I/SAVED/THE/MODEL",
                  load_path=None,
                  logs_dir="/WHERE/I/SAVED/THE/LOGS",
                  training_param=tp,
                  kwargs_converters=kwargs_converters,
                  kwargs_archi=kwargs_archi)
        finally:
            env.close()

    """

    # Limit gpu usage
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except AttributeError:
         # issue of https://stackoverflow.com/questions/59266150/attributeerror-module-tensorflow-core-api-v2-config-has-no-attribute-list-p
        try:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except Exception:
            warnings.warn(_WARN_GPU_MEMORY)
    except Exception:
        warnings.warn(_WARN_GPU_MEMORY)

    if training_param is None:
        training_param = TrainingParam()

    # compute the proper size for the converter
    kwargs_archi["action_size"] = SAC.get_action_size(env.action_space, filter_action_fun, kwargs_converters)

    if load_path is not None:
        path_model, path_target_model = SAC_NN.get_path_model(load_path, name)
        if verbose:
            print("INFO: Reloading a model, the architecture parameters provided will be ignored")
        nn_archi = SAC_NNParam.from_json(os.path.join(path_model, "nn_architecture.json"))
    else:
        nn_archi = SAC_NNParam(**kwargs_archi)

    baseline = SAC(action_space=env.action_space,
                   nn_archi=nn_archi,
                   name=name,
                   istraining=True,
                   verbose=verbose,
                   **kwargs_converters
                   )

    if load_path is not None:
        if verbose:
            print("INFO: Reloading a model, training parameters will be ignored")
        baseline.load(load_path)
        training_param = baseline._training_param

    baseline.train(env,
                   iterations,
                   save_path=save_path,
                   logdir=logs_dir,
                   training_param=training_param)
    # as in our example (and in our explanation) we recommend to save the mode regurlarly in the "train" function
    # it is not necessary to save it again here. But if you chose not to follow these advice, it is more than
    # recommended to save the "baseline" at the end of this function with:
    # baseline.save(path_save)


if __name__ == "__main__":
    # import grid2op
    import numpy as np
    from grid2op.Parameters import Parameters
    from grid2op import make
    from grid2op.Reward import L2RPNReward
    import re
    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()

    args = cli_train().parse_args()

    # is it highly recommended to modify the reward depening on the algorithm.
    # for example here i will push my algorithm to learn that plyaing illegal or ambiguous action is bad
    class MyReward(L2RPNReward):
        def initialize(self, env):
            self.reward_min = 0.0
            self.reward_max = 1.0

        def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
            if has_error or is_illegal or is_ambiguous:
                # previous action was bad
                res = self.reward_min
            elif is_done:
                # really strong reward if an episode is over without game over
                res = self.reward_max
            else:
                res = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
                res /= env.n_line
                if not np.isfinite(res):
                    res = self.reward_min
            return res

    # Use custom params

    # Create grid2op game environement
    env_init = None
    try:
        from grid2op.Chronics import MultifolderWithCache
    except:
        from grid2op.Chronics import MultiFolder
        MultifolderWithCache = MultiFolder

    game_param = Parameters()
    game_param.NB_TIMESTEP_COOLDOWN_SUB = 2
    game_param.NB_TIMESTEP_COOLDOWN_LINE = 2
    env = make(args.env_name,
               param=game_param,
               reward_class=MyReward,
               backend=backend,
               chronics_class=MultifolderWithCache
               )
    # env.chronics_handler.set_max_iter(7*288)
    try:
        env.chronics_handler.real_data.set_filter(lambda x: re.match(".*((03)|(72)|(57))$", x) is not None)
        env.chronics_handler.real_data.reset()
    except RuntimeError as exc_:
        raise exc_
    except AttributeError as exc_:
        # not available in all grid2op version
        pass
    # env.chronics_handler.real_data.
    env_init = env
    if args.nb_env > 1:
        from l2rpn_baselines.utils import make_multi_env
        env = make_multi_env(env_init=env_init, nb_env=int(args.nb_env))

    tp = TrainingParam()

    # NN training
    tp.lr = 1e-4
    tp.lr_decay_steps = 30000
    tp.minibatch_size = 256
    tp.update_freq = 128

    # limit the number of time steps played per scenarios
    tp.step_increase_nb_iter = 100  # None to deactivate it
    tp.min_iter = 10
    tp.update_nb_iter = 100  # once 100 scenarios are solved, increase of "step_increase_nb_iter"

    # oversampling hard scenarios
    tp.oversampling_rate = 3

    # experience replay
    tp.buffer_size = 1000000

    # e greedy
    tp.min_observation = 10000
    tp.initial_epsilon = 0.4
    tp.final_epsilon = 1./(2*7*288.)
    tp.step_for_final_epsilon = int(1e5)

    # don't start always at the same hour (if not None) otherwise random sampling, see docs
    tp.random_sample_datetime_start = None

    # saving, logging etc.
    tp.save_model_each = 10000
    tp.update_tensorboard_freq = 256

    li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                     "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                     "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]

    # nn architecture
    observation_size = SAC_NNParam.get_obs_size(env_init, li_attr_obs_X)
    sizes_q = [800, 800, 800, 494, 494, 494]  # sizes of each hidden layers
    sizes_v = [800, 800]  # sizes of each hidden layers
    sizes_pol = [800, 800, 800, 494, 494, 494]  # sizes of each hidden layers
    kwargs_archi = {'observation_size': observation_size,
                    'sizes': sizes_q,
                    'activs': ["relu" for _ in range(len(sizes_q))],
                    "list_attr_obs": li_attr_obs_X,
                    "sizes_value": sizes_v,
                    "activs_value": ["relu" for _ in range(len(sizes_v))],
                    "sizes_policy": sizes_pol,
                    "activs_policy": ["relu" for _ in range(len(sizes_pol))]
                    }

    # which actions i keep
    kwargs_converters = {"all_actions": None,
                         "set_line_status": False,
                         "change_bus_vect": True,
                         "set_topo_vect": False,
                         }
    nm_ = args.name if args.name is not None else DEFAULT_NAME
    try:
        train(env,
              name=nm_,
              iterations=args.num_train_steps,
              save_path=args.save_path,
              load_path=args.load_path,
              logs_dir=args.logs_dir,
              training_param=tp,
              kwargs_converters=kwargs_converters,
              kwargs_archi=kwargs_archi)
    finally:
        env.close()
        if args.nb_env > 1:
            env_init.close()
