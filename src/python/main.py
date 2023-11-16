import os
from datetime import datetime
import torch
import numpy as np
import imageio
import wandb
from tqdm import tqdm
from PPO import PPO


def train():
    print(
        "============================================================================================"
    )
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(
        project="grid2op",
        notes=current_time,
    )

    ####### initialize environment hyperparameters ######
    env_name = "Grid2OpGeneratorTargetTestEnv"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 100  # max timesteps in one episode
    max_training_timesteps = (
        500000  # break training loop if timeteps > max_training_timesteps
    )

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = (
        max_training_timesteps // 5
    )  # save model frequency (in num timesteps)

    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.95  # discount factor
    entropy_max_loss = 0.0001
    entropy_min_loss = 0.0000001

    lr_actor = 0.00003  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network

    random_seed = 42  # set random seed if required (0 = no random seed)
    #####################################################

    ################# logging variables #################
    wandb.config = {
        "env_name": env_name,
        "has_continuous_action_space": has_continuous_action_space,
        "max_ep_len": max_ep_len,
        "max_training_timesteps": max_training_timesteps,
        "print_freq": print_freq,
        "log_freq": log_freq,
        "save_model_freq": save_model_freq,
        "update_timestep": update_timestep,
        "K_epochs": K_epochs,
        "eps_clip": eps_clip,
        "gamma": gamma,
        "lr_actor": lr_actor,
        "lr_critic": lr_critic,
        "random_seed": random_seed,
    }

    print("training environment name : " + env_name)
    from environment import TestEnv

    env = TestEnv()

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + "/" + env_name + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + "/PPO_" + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    directory = "logs/PPO"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + "/" + env_name + "/"
    checkpoint_dir = directory + "/" + current_time
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print("save checkpoint path : " + checkpoint_dir)
    #####################################################

    ############# print all hyperparameters #############
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print(
        "printing average reward over episodes in last : "
        + str(print_freq)
        + " timesteps"
    )
    print(
        "--------------------------------------------------------------------------------------------"
    )
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
    else:
        print("Initializing a discrete action space policy")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print(
        "============================================================================================"
    )

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(
        env.observation_space,
        env.action_space,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
    )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print(
        "============================================================================================"
    )

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write("episode,timestep,reward\n")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    # training loop
    while time_step <= max_training_timesteps:
        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()
                ppo_agent.entropy_loss_weight = max(
                    entropy_min_loss,
                    entropy_max_loss
                    - (entropy_max_loss - entropy_min_loss)
                    * time_step
                    / max_training_timesteps,
                )

            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(float(log_avg_reward), 4)

                log_f.write("{},{},{}\n".format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(float(print_avg_reward), 2)

                print(
                    "Episode : {} \t\t Timestep : {} \t\t Average Reward : {} Entropy Loss weight: {}".format(
                        i_episode,
                        time_step,
                        print_avg_reward,
                        ppo_agent.entropy_loss_weight,
                    )
                )

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print(
                    "--------------------------------------------------------------------------------------------"
                )
                curr_dir = checkpoint_dir + "/time_step_" + str(time_step)
                if not os.path.exists(curr_dir):
                    os.makedirs(curr_dir)
                checkpoint_path = curr_dir + "/PPO_{}_{}_{}.pth".format(
                    env_name, random_seed, time_step
                )
                print("Saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                draw_agent(env, ppo_agent, curr_dir)
                print("Model saved")
                print(
                    "Elapsed Time  : ",
                    datetime.now().replace(microsecond=0) - start_time,
                )
                print(
                    "--------------------------------------------------------------------------------------------"
                )

            # break; if the episode is over
            if done:
                break
        wandb.log({"reward": current_ep_reward, "timestep": time_step})
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print(
        "============================================================================================"
    )
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print(
        "============================================================================================"
    )

    agent_final_checkpoint = f"PPO_{env_name}_{random_seed}_{time_step}.pth"
    ppo_agent.save(os.path.join(wandb.run.dir, agent_final_checkpoint))  # type: ignore
    wandb.finish()

    print("Training Done!")


def draw_agent(env, ppo_agent: PPO, checkpoint_path):
    print("Start drawing agent")
    obs, _ = env.reset(seed=1)
    frames = []
    rewards = []
    done = False
    for i in tqdm(range(100)):
        action = ppo_agent.select_action_eval(obs)
        obs, reward, done, terminated, _ = env.step(action)
        # print(obs[:, 0])
        rewards.append(reward)
        frames.append(env.render(mode="rgb_array"))
        done = done or terminated
    imageio.mimsave(checkpoint_path + "/movie.gif", frames)
    print("RL Reward:", sum(rewards))
    print("Done")


if __name__ == "__main__":
    train()
