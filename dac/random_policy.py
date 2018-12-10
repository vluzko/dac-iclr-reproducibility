"""Run a random policy on the given environment"""
import gym
import fire
import pandas as pd

import config


def run_random_policy(env_id="Hopper-v1", evaluations=1000, trajectories=3):
    """

    Args:
        env_id (str): The environment to evaluate on.
        evaluations (int): The number of evaluations.
        trajectories (int): The number of trajectories to average over.

    Returns:

    """
    env = gym.make(env_id)
    rewards = []
    for i in range(evaluations + 1):
        traj_rewards = []
        for _ in range(trajectories):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                _, reward, done, _ = env.step(env.action_space.sample())
                total_reward += reward
            traj_rewards.append(total_reward)
        traj_rewards.append(i * 1000)
        rewards.append(traj_rewards)

    df = pd.DataFrame.from_records(rewards)
    columns = ["reward_{}".format(i) for i in range(trajectories)]
    columns.append("timestep")
    df.columns = columns

    results_fname = 'Random_{}.csv'.format(env_id)
    df.to_csv(str(config.results_dir / results_fname))


if __name__ == "__main__":
    fire.Fire(run_random_policy)
