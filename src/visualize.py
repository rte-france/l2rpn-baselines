from ray.rllib.algorithms.algorithm import Algorithm
import env_0
import agent_0
import imageio
from tqdm import tqdm

my_new_ppo = Algorithm.from_checkpoint("/home/scheschb/ray_results/PPO_2023-11-02_16-11-08/PPO_test_env_1afdb_00000_0_2023-11-02_16-11-08/checkpoint_000009")
env = env_0.TestEnv()

obs, _ = env.reset()
frames = []
rewards = []
for i in tqdm(range(100)):
    action = my_new_ppo.compute_single_action(obs)
    obs, reward, done, terminated, info = env.step(action)
    rewards.append(reward)
    frames.append(env.render(mode="rgb_array"))
    if done:
        break
print("RL Reward:", sum(rewards))
imageio.mimsave("movie.gif", frames)
print("Done")

obs, _ = env.reset()
frames = []
rewards = []
for i in tqdm(range(100)):
    # action = my_new_ppo.compute_single_action(obs)
    action = -2.0*(obs[0]>=0)+1.0
    obs, reward, done, terminated, info = env.step(action)
    rewards.append(reward)
    frames.append(env.render(mode="rgb_array"))
    if done:
        break
imageio.mimsave("movie_expert.gif", frames)
print("Expert Reward:", sum(rewards))
print("Done")
