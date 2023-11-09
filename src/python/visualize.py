from ray.rllib.algorithms.algorithm import Algorithm
import env
import agent
import imageio
import argparse
from tqdm import tqdm
import ray

if __name__ == "__main__":
    parsers = argparse.ArgumentParser()
    parsers.add_argument(
        "--checkpoint",
        type=str,
        default="/home/scheschb/ray_results/PPO_2023-11-08_14-45-52/PPO_test_env_9205e_00000_0_2023-11-08_14-45-52/checkpoint_000035",
    )
    args = parsers.parse_args()

    ray.init()
    my_new_ppo = Algorithm.from_checkpoint(args.checkpoint)
    env = env.TestEnv(env_name="l2rpn_case14_sandbox")

    obs, _ = env.reset(seed=0)
    frames = []
    rewards = []
    for i in tqdm(range(100)):
        action = my_new_ppo.compute_single_action(obs, explore=False, update=False)
        obs, reward, done, terminated, info = env.step(action)
        # print(obs[:, 0])
        rewards.append(reward)
        frames.append(env.render(mode="rgb_array"))
        if done:
            break
    imageio.mimsave("movie.gif", frames)
    print("RL Reward:", sum(rewards))
    print("Done")

    # obs, _ = env.reset(seed=42)
    # frames = []
    # rewards = []
    # for i in tqdm(range(100)):
    #     action = -2.0 * (obs[:, 0] >= 0) + 1.0
    #     obs, reward, done, terminated, info = env.step(action)
    #     rewards.append(reward)
    #     frames.append(env.render(mode="rgb_array"))
    #     if done:
    #         break
    # imageio.mimsave("movie_expert.gif", frames)
    # print("Expert Reward:", sum(rewards))
    # print("Done")
