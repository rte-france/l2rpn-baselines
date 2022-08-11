# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import shutil
from typing import Union

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import grid2op
import numpy as np
import pkg_resources
from grid2op.Agent import BaseAgent
from tensorflow.python.training.tracking.tracking import AutoTrackable
from ubelt import Timer

pkg_resources.require("tensorflow>=2.7.0")
pkg_resources.require("keras>=2.7.0")
import tensorflow as tf

from curriculumagent.senior.rllib_execution.convert_rllib_ckpt import load_and_save_model, load_config
from curriculumagent.teacher.collect_teacher_experience import make_unitary_actionspace
from curriculumagent.teacher.teacher import general_teacher
from curriculumagent.tutor.collect_tutor_experience import prepare_dataset
from curriculumagent.tutor.tutors.general_tutor import generate_tutor_experience
from curriculumagent.junior.junior_student import train as train_junior
from curriculumagent.senior.rllib_execution.senior_env_rllib import SeniorEnvRllib
from curriculumagent.senior.rllib_execution.senior_model_rllib import Grid2OpCustomModel
from curriculumagent.submission.my_agent_advanced import MyAgent


class CurriculumAgent(MyAgent):
    """
    This is the Baseline of the Curriculum Agent, which in turn is a refurbished agent based on the submission of
    @https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution. Their work was published under a Mozilla Public
    Licence, so in turn, please credit their work as well, when using this code.

    The overall agent follows the typical baseline format. However, given that the original code consists of four
    different sub-agents, this agent inherits this complexity as well. Thus, the whole execution of the training
    pipeline can be found in the train() method of the agent. Given that four different agents have to be trained,
    and the teacher + tutor agents are based on simply gready searches, the overall training might take some time.

    The default training results were created for the IEEE14 of the grid2op package, "named l2rpn_case14_sandbox".

    """

    def __init__(self, action_space, name, model_path: Union[Path, str], **kwargs):
        """ Init Mehtod of the Curriculum Agent

        Args:
            action_space: action space of the Grid2Op Env
            observation_space: observation space of the Grid2Op Env
            name: Name of the agent.
            model_path: Path, where to find the dirs of the different agents. Note, this should include a
            teacher, tutor, junior and senior directory in order to both load the action space of the
            teacher as well as the model of the senior agent.
            **kwargs: No additional kwargs are expected.
        """

        if isinstance(model_path, Path):
            self.path_of_data = model_path
        else:
            self.path_of_data = Path(model_path)

        self.action_space = action_space

        action_space_file = self.path_of_data/ "agent" / "actions"/"CA_actions.npy"
        model_path = self.path_of_data / "agent" / "model"

        assert action_space_file.is_file(), f"Action space file {action_space_file} does not exists"
        assert model_path.is_dir(), f"Model dir {model_path} does not exists."
        self.model_path = model_path
        self.action_path = action_space_file

        logging.info(f"Try loading agent from {self.path_of_data}.")

        self.agent = MyAgent(action_space=self.action_space,
                             model_path=model_path,
                             action_space_path=action_space_file,
                             filtered_obs=False)

        self.name = name

    def act(self, observation, reward, done) -> grid2op.Action.BaseAction:
        """ This is the act function of the Curriculum Agent.

        Args:
            observation: grid2op Observation
            reward: Reward of previous step
            done: Whether the agent completed the episode

        Returns: Grid2op Action

        """
        action = self.agent.act(observation=observation, reward=reward, done=done)
        return action

    def reset(self, observation):
        """
        This method is called at the beginning of a new episode.
        It is implemented by baselines to reset their internal state if needed.
        This method is not required for the curriculum agent.

        Args:
            observation: Observation of the Grid2Op Environment

        Returns: None

        """
        pass

    def load(self, path: Union[str, Path]):
        """
        If wanted, you can load the Curriculum Agent from a different file.

        In order to have all files, you at least need the teacher and the agent directory with the reduced
        action space and the saved model. In order to gather the correct input, please run the train function in
        advance.


        Args:
            path: Path to the trained CurriculumAgent

        Returns: None

        """

        if not isinstance(path, (str, Path)):
            raise ValueError("Not the correct variable type in path variable.")

        if isinstance(path, str):
            path = Path(path)


        action_space_file = path / "agent" / "actions" / "CA_actions.npy"
        model_path = path / "agent" / "model"

        assert action_space_file.is_file(), f"Action space file {action_space_file} does not exists"
        assert model_path.is_dir(), f"Model dir {model_path} does not exists."
        self.model_path = model_path
        self.action_path = action_space_file
        logging.info("Overwriting internal model and action path.")

        self.agent = MyAgent(action_space=self.action_space,
                             model_path=model_path,
                             action_space_path=action_space_file,
                             filtered_obs=False)
        logging.info("Loading the agent fulfilled.")

    def save(self, path:Union[str, Path]):
        """ The save method creates two directories. First the directory where to finde the
        Args:
            path: Path, where to save the model

        Returns: None

        """
        if not isinstance(path, (str, Path)):
            raise ValueError("Not the correct variable type in path variable.")

        if isinstance(path, str):
            path = Path(path)

        assert path.is_dir(), f"The directory {path} does not exists. "

        [(path / "agent" / n).mkdir(exist_ok=True, parents=True) for n in ["model","actions"]]
        action_space_file = path / "agent" / "actions" / "CA_actions.npy"

        # Write actions:
        np.save(action_space_file,self.agent.actions)

        # Write model:
        model_path = path / "agent" / "model"
        if isinstance(self.agent.model,AutoTrackable):
            logging.info(f"Write TF model of type{type(self.agent.model)}")
            tf.saved_model.save(self.agent.model,str(model_path ))
        else:
            logging.info("Try writing Keras Model")
            self.agent.model.save( model_path )

    def train(self,
              env: Union[str, grid2op.Environment.BaseEnv] = "l2rpn_case14_sandbox",
              name="Example",
              iterations=1,
              save_path: Optional[Path] = None,
              load_path: Optional[Path] = None,
              **kwargs) -> grid2op.Agent.BaseAgent:
        """
        This function is used to train the Curriculum Agent. For this please provide the environment, either as a
        string or as the BaseEnvironment file.

        Note, this might take a while.

        Args:
            env: Str, Path to environment or the grid2op Environment.
            iterations: Number of iterations for the training
            save_path: Optional save_path argument, if you want to store your agent somewhere else.
            **kwargs: Optional Parameters such as seed , jobs, number of trials or number of gpus. Default values are
                        {"seed":2,"jobs": os.cpu_count(),"num_trials":1,"num_gpus":0}

        Returns: Returns the fitted agent.

        """
        base_config = {"seed": 2,
                       "jobs": os.cpu_count(),
                       "num_trials": 1,
                       "num_gpus": 0}

        # Check for kwargs:
        for k,v in kwargs.items():
            if k in base_config.keys() and isinstance(v,int):
                base_config[k] = v

        # Set seed:
        np.random.seed(base_config["seed"])

        # Prepare environment
        log_format = '(%(asctime)s) [%(name)-10s] %(levelname)8s: %(message)s [%(filename)s:%(lineno)s]'
        logging.basicConfig(level=logging.INFO, format=log_format)

        # Make sure environment is present or downloaded.
        # For parallelization, this needs to be only the name
        if isinstance(env, grid2op.Environment.BaseEnv):
            env_name = Path(env.chronics_handler.path).parent
        elif Path(env).is_dir():
            env_name = env
        elif env not in grid2op.list_available_local_env():
            logging.info(f"{env} was not downloaded yet. Trying to download...")
            env_name = env
            grid2op.make(env)
        elif isinstance(env, str):
            env_name = env
        else:
            logging.info("The provided env argument does not correspond with the expected format. Reverting to "
                         "l2rpn_case14_sandbox")
            env_name = "l2rpn_case14_sandbox"

        if name is None:
            # Use current time if run name is not given
            now = datetime.now().strftime("%d%m%Y_%H%M%S")
            name = now

        # Create save path if not submitted:
        if isinstance(save_path, Path):
            run_path = save_path
        # Fallback to the load path:
        elif isinstance(load_path, Path):
            run_path = save_path
        else:  #
            if isinstance(env, grid2op.Environment.BaseEnv):
                run_path = Path(f"curriculumagent_train_{env.name}_{name}").absolute()

            else:
                run_path = Path(f"curriculumagent_train_{env}_{name}").absolute()
            run_path.mkdir(exist_ok=True, parents=True)
            logging.info(f"Creating directory{run_path}")
        if not run_path.exists():
            run_path.mkdir(exist_ok=True, parents=True)
            logging.info(f"Creating directory{run_path}")

        for sub_dir in ["teacher", "tutor", "junior", "senior", "agent"]:
            (run_path / sub_dir).mkdir(exist_ok=True, parents=True)

        total_timer = Timer()
        total_timer.tic()

        # Stage1 : Teacher
        n_episodes = iterations
        limit_chronics = 42
        max_actionspace_size = 250
        # Run Teacher / Do Action Space Reduction
        teacher_experience_path = run_path / "teacher" / Path("general_teacher_experience.csv")
        if not teacher_experience_path.exists():
            logging.info(f"Finding good actions using Teacher...")
            general_teacher(save_path=teacher_experience_path, env_name_path=env_name,
                            limit_chronics=limit_chronics, n_episodes=n_episodes, jobs= base_config["jobs"])
        else:
            logging.info(f"Skipping Teacher because {teacher_experience_path} already exists")

        # Use experience to do action space reduction
        reduced_action_space_path = run_path / "teacher" / Path("reduced_actionspace.npy")
        if not reduced_action_space_path.exists():
            logging.info("Reducing actionspace...")
            make_unitary_actionspace(reduced_action_space_path, [teacher_experience_path], env_name,
                                     best_n=max_actionspace_size)
        else:
            logging.info(f"Skipping action space generation because {reduced_action_space_path} already exists")

        # Stage 2 : Tutor
        tutor_experience_path = run_path / "tutor" / "tutor_experience.npy"
        if not tutor_experience_path.exists():
            logging.info("Generating experience using Tutor...")
            generate_tutor_experience(env_name, tutor_experience_path, reduced_action_space_path,
                                      num_chronics=10 * iterations, jobs=base_config["jobs"], seed=base_config["seed"])
        else:
            logging.info(f"Skipping Tutor because {tutor_experience_path} already exists")

        # Stage 3: Junior

        junior_data_path = run_path / "tutor" / "junior_data"
        junior_results_path = run_path / "junior"
        junior_data_path.mkdir(exist_ok=True, parents=True)
        prepare_dataset(traindata_path=tutor_experience_path.parent,
                        target_path=junior_data_path,
                        filtered_obs=False,
                        dataset_name="test")

        # print(junior_data_path)
        if not (junior_results_path / "saved_model.pb").exists():
            train_junior(run_name="junior",
                         dataset_path=junior_data_path,
                         target_model_path=junior_results_path,
                         action_space_file=reduced_action_space_path,
                         dataset_name='test',
                         epochs=100 * iterations, seed=42)
        else:
            logging.info(f"Skipping Junior because {junior_results_path / 'saved_model.pb'} already exists")

        # Senior:
        # Configure configs & register model:
        # Given some import problems we only import ray here:
        senior_results_path = run_path / "senior"
        agent_path = run_path / "agent"

        if not (senior_results_path / "sandbox").exists():
            import ray
            from ray import tune
            from ray.rllib.models import ModelCatalog
            ray.init()
            env_config = {"action_space_path": reduced_action_space_path,
                          "env_path": env_name,
                          "action_threshold": 0.9,
                          'filtered_obs': False}
            model_config = {"path_to_junior": junior_results_path}
            ModelCatalog.register_custom_model('binbinchen', Grid2OpCustomModel)

            num_workers = int(np.floor((base_config["jobs"] - base_config["num_trials"]) / base_config["num_trials"]))
            # Run Ray:
            tune.run(
                "PPO",
                name="sandbox",
                checkpoint_freq=1,
                keep_checkpoints_num=10,
                verbose=1,
                max_failures=1,
                num_samples=1,  # Adjust number of samples accordingly
                local_dir=senior_results_path,
                stop={"training_iteration": 10 * iterations},
                config={
                    "env": SeniorEnvRllib,
                    "env_config": env_config,
                    "num_workers": num_workers,
                    "lr": 5e-5,
                    "num_gpus":  base_config["num_gpus"] /  base_config["num_trials"],
                    "num_cpus_per_worker": 1,
                    "remote_worker_envs": False,
                    "model": {"custom_model": "binbinchen", "custom_model_config": model_config},
                },
            )
            # Now load and save the model in tensorflow readable format.
            # Rename the created checkpoint. Normally this should only be one !

            training_dir_names = [name for name in os.listdir(senior_results_path / "sandbox")
                                  if "PPO_SeniorEnvRllib" in name]

            config, ckpt_path = load_config(senior_results_path / "sandbox" / training_dir_names[-1], latest=True)
            config['env_config'] = env_config
            config["model"]['custom_model_config']['custom_config'] = model_config

            # Now lets load the agent:
            load_and_save_model(ckpt_path=ckpt_path, config=config,
                                save_path=agent_path)

            ray.shutdown()
        else:
            logging.info(f"Skipping Senior because {senior_results_path / 'sandbox'} already exists")

        if isinstance(env, grid2op.Environment.BaseEnv):
            grid_env = env
        else:
            grid_env = grid2op.make(env)

        # Copy Action Space path to agent path:
        action_space_path = run_path/"agent"/"actions"
        action_space_path.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(reduced_action_space_path, action_space_path / "CA_actions.npy")
        if (action_space_path / "CA_actions.npy").exists():
            logging.info(f"Write actions of agent to {action_space_path /'CA_actions.npy'}")


        self.agent = MyAgent(action_space=grid_env.action_space,
                             model_path=run_path / "agent" / "model",
                             this_directory_path=run_path,
                             action_space_path=action_space_path,
                             filtered_obs=False,
                             scaler=None)

        logging.info(f"Loading of Agent Succeeded. Return MyAgent.")

        total_timer.toc()
        logging.info(f"Pipeline execution took {total_timer.elapsed}")
        return self.agent


if __name__ == '__main__':
    env = grid2op.make("l2rpn_case14_sandbox")
    obs = env.reset()
    path_of_model = Path(env.chronics_handler.path).parent

    myagent = CurriculumAgent(
        action_space=env.action_space,
        model_path=Path(__file__).parent/"model_IEEE14",
        name="Test")

    obs = env.reset()
    done = False
    while not done:
        act = myagent.act(observation=obs, reward=0, done=False)
        obs, rew, done, info = env.step(act)
    print(f"The baseline survived {env.nb_time_step} timesteps")