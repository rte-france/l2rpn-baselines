# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Union, List, Optional, Tuple

import grid2op
import numpy as np
import ray
from grid2op.Agent import BaseAgent
from lightsim2grid import LightSimBackend

from curriculumagent.junior.junior_student import train as train_junior
from curriculumagent.senior.senior_student import Senior
from curriculumagent.submission.my_agent import MyAgent
from curriculumagent.teacher.collect_teacher_experience import make_unitary_actionspace
from curriculumagent.teacher.teacher import general_teacher
from curriculumagent.tutor.collect_tutor_experience import generate_tutor_experience, prepare_dataset


class CurriculumAgent(BaseAgent):
    """
    This is the Baseline of the Curriculum Agent, which in turn is a refurbished agent based on the submission of
    @https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution. Their work was published under a Mozilla Public
    Licence, so in turn, please credit their work as well, when using this code.

    The overall agent follows the typical baseline format. However, given that the original code consists of four
    different sub-agents, we decided to separate the agents in order to make it easier to understand.
    With the full pipeline, the CurriculumAgent consist of a Teacher,a Tutor, a Junior and a Senior agent that
    all contribute to the final MyAgent. In order to train the full pipeline, please execute the train_full_pipeline()
    method. Given that four different agents have to be trained,
    and the teacher + tutor agents are based on simply gready searches, the overall training might take some time.

    However, if you just want to use the agent or retrain the Senior (RL-Model), you can use the train method.
    The default training results were created for the IEEE14 of the grid2op package, named "l2rpn_case14_sandbox".

    """

    def __init__(self,
                 action_space: grid2op.Action.ActionSpace,
                 observation_space: grid2op.Observation.BaseObservation,
                 name: str,
                 **kwargs):
        """ Init Method of the CurriculumAgent. Note that the agent has to be loaded after initilization in order to
        work.

        Args:
            action_space: Action_space of Grid2Op Environment
            observation_space: Observation Space of the Grid2Op Environment
            name: Name of the Agent

            **kwargs: Additional values that might be helpfull when loading the MyAgent
        """
        BaseAgent.__init__(self, action_space)
        self.name = name
        self.observation_space = observation_space
        self.agent: MyAgent = None
        self.ckpt_path = None
        self.do_nothing = self.action_space({})
        self.name = name
        self.senior: Senior = None

        self.__kwargs = {}
        if any(kwargs.keys()):
            self.__kwargs = kwargs

    def act(self, observation: grid2op.Observation.BaseObservation,
            reward: float, done: bool) -> grid2op.Action.BaseAction:
        """ This is the act function of the CurriculumAgent.

        Args:
            observation: Grid2Op Observation
            reward: Reward of the previous step
            done: Whether the environment is done.

        Returns: Action of the agent

        """
        if self.agent is None:
            warnings.warn("The agent was not yet loaded, thus we return the do-nothing action. "
                          "Please run the load() for the CurriculumAgent.")
            action = self.do_nothing
        else:
            action = self.agent.act(observation=observation, reward=reward, done=done)
        return action

    def reset(self, observation: grid2op.Observation.BaseObservation) -> None:
        """
        This method is called at the beginning of a new episode.
        It is implemented by baselines to reset their internal state if needed.
        This method is not required for the CurriculumAgent.

        Args:
            observation: Observation of the Grid2Op Environment

        Returns: None

        """
        self.agent.reset(observation)

    def load(self, path: Union[str, Path],
             actions_path: Optional[Union[Path, List[Path]]] = None,
             **kwargs) -> None:
        """This method loads the MyAgent and inits the agent.

        We provide three ways to load the agent:
        1. You provide the Senior model path (preferred) of a trained Senior model
        2. You can provide a Junior model path, prior to the training.


        Note:
            We load with the default values, if you want to pass specific values to the MyAgent method,
            you can do that via kwargs. These can be for example the boolean subset (whether you
            do not want the whole observation) or the boolean topo (whether you want to revert your topology).
            Depending on the underlying model, you might need to train with the kwarg: {"subset":True}

        Args:
            path: path, where we can find the model for execution
            actions_path: path, where to find the action sets. This is required to run the agent. If this is not
            provided, we try to load from the same directory as the model.
            kwargs: You can pass further arguments to the MyAgent

        Returns:
            None.

        """
        # Check both input paths:
        action_path, model_path = self.__check_paths(path,
                                                     actions_path)

        if any(kwargs.keys()):
            for k, v in kwargs.items():
                self.__kwargs[k] = v

        self.agent = MyAgent(action_space=self.action_space,
                             model_path=model_path,
                             action_space_path=action_path,
                             **self.__kwargs)

    def save(self, path: Union[str, Path]) -> None:
        """ This method saves the model of the agent. Note that this requires the agent to
        be initialized

        Args:
            path: Where to save the model

        Returns: None

        """
        if self.agent is None:
            raise ValueError("Model can not be saved, because there is no agent. Please initialize the agent.")

        # Create paths:

        model_path = Path(path) / "model"
        if not model_path.is_dir():
            os.mkdir(model_path)
        self.agent.model.save(model_path)
        logging.info(f"Model is saved under {model_path}")

        action_path = Path(path) / "actions"
        if not action_path.is_dir():
            os.mkdir(action_path)
        np.save(action_path / "actions.npy", self.agent.actions)
        logging.info(f"Actions are saved under {action_path / 'actions.npy'}")

    def train(self, env: Union[grid2op.Environment.BaseEnv, str] = "l2rpn_case14_sandbox",
              iterations: int = 1,
              save_path: Union[Path, str] = Path(os.getcwd()) / "out",
              **kwargs_senior):
        """
        For the training, we initialize the Senior student and then run the training

        Note that in order for this to work, we need Ray. If it is not initialized, we
        call ray.init(). Accordingly, if you have specific cluster resources, make sure to
        call ray.init() prior to training this agent!

        Args:
            env: Grid2Op Environment
            iterations: Number of iterations to run the senior
            save_path: Where to save the model and the rllib checkpoints.
            **kwargs_senior: Additional arguments for the Senior model

        Returns: MyAgent

        """
        if isinstance(env, grid2op.Environment.BaseEnv):
            env_path = env.get_path_env()
        else:
            env_path = env

        # First we save the current model to ensure that we have the correct actions:
        assert self.agent is not None, "The agent has not been loaded. Please load the agent first!"
        self.save(save_path)

        # Test if files are there:
        action_path, model_path = self.__check_paths(save_path)

        # Gather all variables:
        ckpt_path = save_path / "ckeckpoints"
        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        logging.info("Saving previous model, action and checkpoint directory done!")
        # Ray:
        if not ray.is_initialized:
            ray.init()

        # Train the senior
        self.senior = Senior(env_path=env_path,
                             action_space_path=action_path,
                             model_path=model_path,
                             ckpt_save_path=ckpt_path,
                             **kwargs_senior)

        logging.info(f"Initiation of Senior done, let's start training with {iterations}.")
        self.senior.train(iterations=iterations)
        logging.info(f"Training complete. Overwriting agent")
        self.agent = self.senior.get_my_agent(path=model_path)
        logging.info(f"Completion of training")

    def train_full_pipeline(
            self,
            env: Union[str, grid2op.Environment.BaseEnv] = "l2rpn_case14_sandbox",
            iterations: int = 1,
            save_path: Optional[Path] = Path(os.getcwd()) / "out",
            **kwargs,
    ) -> grid2op.Agent.BaseAgent:
        """
        This method is used to train the full pipeline of the Curriculum Agent,i.e., the Teacher-Tutor-Junio-Senior
        framework.

        This means, that after initializing all directories, the Teacher will search for suitable
        actions of the Grid2Op environment. Afterwards, the Tutor will take a subset of these actions
        and generate an action-observation experience set. Then, the Junior (feed-forward) model, will
        learn to imitat the Tutor. As a last step the Senior receives the weights of the junior and
        learns with Reinforcement Learning to navigate in the environment.

        Note, it is necessary to provide a suiting Grid2Op environment. Further, expect that this might
        take a while.

        Args:
            env: Str, Path to environment or the grid2op Environment.
            iterations: Number of iterations for the training
            save_path: Optional save_path argument, if you want to store your agent somewhere else.
            **kwargs: Optional Parameters such as seed or jobs, or max_action_space_size

        Returns: Returns the fitted agent.

        """
        # Check for kwargs:
        seed = kwargs.get("seed", 42)
        jobs = kwargs.get("jobs", os.cpu_count())
        max_actionspace_size = kwargs.get("max_actionspace_size", 250)
        # Set seed:
        np.random.seed(seed)

        # Prepare environment
        log_format = "(%(asctime)s) [%(name)-10s] %(levelname)8s: %(message)s [%(filename)s:%(lineno)s]"
        logging.basicConfig(level=logging.INFO, format=log_format)

        # Make sure environment is present or downloaded.
        # For parallelization, this needs to be only the name
        if isinstance(env, grid2op.Environment.BaseEnv):
            env_path = env.get_path_env()
        else:
            env_path = env

        try:
            grid2op.make(env_path, backend=LightSimBackend())
        except Exception as e:
            raise ValueError(f"Could not init the Grid2op Enviornment with LightSimBackend. Please ensure that" \
                             f"this works! We raise a {e}")

        for sub_dir in ["teacher", "tutor", "junior", "senior", "model", "actions"]:
            (save_path / sub_dir).mkdir(exist_ok=True, parents=True)

        logging.info(f"All paths were created under {save_path}. Now let's start with the Teacher training.")

        ############################################################################################
        # Stage1 : Teacher
        # Run Teacher / Do Action Space Reduction
        teacher_experience_path = save_path / "teacher" / Path("general_teacher_experience.csv")
        if not teacher_experience_path.exists():
            logging.info(f"Finding good actions using Teacher...")
            general_teacher(
                save_path=teacher_experience_path,
                env_name_path=env_path,
                n_episodes=iterations,
                jobs=jobs,
            )
        else:
            logging.info(f"Skipping Teacher because {teacher_experience_path} already exists")

        # Use experience to do action space reduction and save as our action space file
        actions_path = save_path / "actions" / "actions.npy"
        if not actions_path.exists():
            logging.info("Reducing actionspace...")
            make_unitary_actionspace(
                actions_path, [teacher_experience_path], env_path,
                best_n=max_actionspace_size
            )
        else:
            logging.info(f"Skipping action space generation because {actions_path} already exists")

        ############################################################################################
        # Stage 2 : Tutor
        tutor_experience_path = save_path / "tutor" / "tutor_experience.npy"
        if not tutor_experience_path.exists():
            logging.info("Generating experience using Tutor...")
            generate_tutor_experience(
                env_path,
                tutor_experience_path,
                actions_path,
                num_chronics=iterations,
                jobs=jobs,
                seed=seed,
            )
        else:
            logging.info(f"Skipping Tutor because {tutor_experience_path} already exists")

        ############################################################################################
        # Stage 3: Junior
        junior_data_path = save_path / "tutor" / "junior_data"
        junior_results_path = save_path / "junior"
        junior_data_path.mkdir(exist_ok=True, parents=True)
        prepare_dataset(
            traindata_path=tutor_experience_path.parent,
            target_path=junior_data_path,
            dataset_name="test",
        )

        if not (junior_results_path / "saved_model.pb").exists():
            train_junior(
                run_name="junior",
                dataset_path=junior_data_path,
                target_model_path=junior_results_path,
                action_space_file=actions_path,
                dataset_name="test",
                epochs=10 + iterations,
                seed=seed,
            )
        else:
            logging.info(f"Skipping Junior because {junior_results_path / 'saved_model.pb'} already exists")

        ############################################################################################
        # Stage 4: Senior:
        # Configure configs & register model:
        # Given some import problems we only import ray here:
        senior_results_path = save_path / "senior"
        if ray.is_initialized() is False:
            ray.init()

        # get the resources:
        resources = ray.nodes()
        num_workers = int(resources[0]["Resources"]["CPU"] // 2)

        senior = Senior(env_path=env_path,
                        action_space_path=actions_path,
                        model_path=junior_results_path,
                        ckpt_save_path=senior_results_path,
                        num_workers=num_workers,
                        subset=False)

        if not (senior_results_path / "sandbox").exists():
            senior.train(iterations=iterations)
        else:
            logging.info(f"Skipping Senior training because {senior_results_path / 'sandbox'} already "
                         f"exists. Instead we are loading the senior from checkpoints")  #
            senior.restore(senior_results_path)
            logging.info(f"The Senior trained for {senior.ppo.iteration} iterations.")

        ############################################################################################
        # Stage 5.
        # Saving the agent

        agent_path = save_path / "model"
        logging.info(f"Saving the agent model and building the agent")
        self.agent = senior.get_my_agent(path=agent_path)
        logging.info(f"Loading of Agent Succeeded. Return MyAgent.")
        ray.shutdown()
        return self.agent

    def __check_paths(self, path: Union[str, Path],
                      action_path: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
        """ Check whether the path is in the required format

        Args:
            path: Path, where the agent should be saved
            action_path: Optional path for the actions, if they were specified differently.

        Returns:

        """
        path = Path(path)
        assert path.is_dir(), "The provided path does not exist. Please pass an existing directory."

        # Action Paths:
        if action_path is None:
            action_path = path / "actions"

        if not isinstance(action_path, list):
            if not action_path.is_dir():
                warnings.warn(f"We could not find the directory{action_path}. We try to import directly from {path}.")
                action_path = path

            if not any([".npy" in a_p for a_p in os.listdir(action_path)]):
                raise FileNotFoundError(f"No .npy was found in the action path {action_path}. Thus, we "
                                        f"can not load the actions")
        else:
            if not any([".npy" in str(a_p) for a_p in action_path]):
                raise FileNotFoundError(f"No .npy was found in the action path list {action_path}. Thus, we "
                                        f"can not load the actions")

        # Model Paths
        model_path = path / "model"
        if not model_path.is_dir():
            warnings.warn(f"We could not find the directory{model_path}. We try to import directly from {path}.")
            model_path = path

        if not any(["saved_model.pb" in a_p for a_p in os.listdir(model_path)]):
            raise FileNotFoundError(f"We could not find any tensorflow model in the directory {model_path}. "
                                    f"Please provide the correct path.")

        return action_path, model_path

    def create_submission(self, path: Union[str, Path] = "./submission"):
        """ Method that takes a saved model and agent and creates a full submission folder in the provided path.

        This only covers the "basic" CurriculumAgent!

        Note that you use a scaler, you have to change the make agent method and add the path. Same holds
        for other configurations.

        Args:
            path: Path where to save the submission

        Returns:
            None

        """
        assert path.is_dir(), "The provided path does not exist. Please pass an existing directory."

        # Save the model in the path:
        self.save(Path(path))

        # Creating directories
        path_for_files = Path(__file__).parent.parent
        path_for_common = Path(path) / "common"

        if not path_for_common.is_dir():
            os.mkdir(path_for_common)

        # Now copying the files from the commons method:
        shutil.copy(path_for_files / "common" / "__init__.py", path_for_common)
        shutil.copy(path_for_files / "common" / "obs_converter.py", path_for_common)
        shutil.copy(path_for_files / "common" / "utilities.py", path_for_common)

        # Move agent
        shutil.copy(path_for_files / "submission" / "my_agent.py", path)
        shutil.copy(path_for_files / "submission" / "__init__.py", path)

        logging.info(f"Moved required files to {path} .")
        logging.info("Note, if you have specific settings (scaler, subset, ...) you have to manually change the "
                     "my_agent method.")
