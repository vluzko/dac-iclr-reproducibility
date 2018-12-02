from util.learning_rate import LearningRate
from util.replay_buffer import ReplayBuffer
from networks.adversary import Discriminator
from networks.TD3 import TD3
from dataset.mujoco_dset import Mujoco_Dset

import gym
import argparse
import numpy as np
import pandas as pd
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def argsparser():
	parser = argparse.ArgumentParser("DAC")
	parser.add_argument('--env_id', help='environment ID', default='Hopper-v1')
	parser.add_argument('--seed', help='RNG seed', type=int, default=0)
	parser.add_argument('--expert_path', type=str, default='trajs/trajs_hopper.h5')
	parser.add_argument('--loss', type=str, default='sum')
	parser.add_argument('--traj_num', help='Number of Traj', type=int, default=4)
	return parser.parse_args()


def main(cl_args):
	# Create the environment to train on.
	env = gym.make(cl_args.env_id)
	sum_or_mean_loss = (cl_args.loss == 'sum')

	# They state they use a batch size of 100 and trajector length of 100 in the OpenReview comments.
	# https://openreview.net/forum?id=Hk4fpoA5Km&noteId=HyebhMXa2X
	# Trajectory length == T in the pseudo-code
	trajectory_length = 1000
	batch_size = 100

	# Train for 1 million timesteps. See Figure 4.
	num_steps = 10000

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	lr = LearningRate.get_instance()
	lr.set_learning_rate(10 ** (-3))  # Loss is 10e-3
	lr.set_decay(1.0 / 2.0)  # Decay is 1/2

	# The buffer for the expert -> refer to dataset/mujoco_dset.py
	expert_buffer = Mujoco_Dset(env, cl_args.expert_path, cl_args.traj_num)
	actor_replay_buffer = ReplayBuffer(env)

	# TD3(state_dim, action_dim, max_action, actor_clipping, decay_steps*) *Not used yet;
	td3_policy = TD3(state_dim, action_dim, max_action, 40, 10 ** 5)

	# Input dim = state_dim + action_dim
	discriminator = Discriminator(state_dim + action_dim).to(device)

	# For storing temporary evaluations
	evaluations = [
		(evaluate_policy(cl_args.env_id, td3_policy, 0), 0)
	]

	evaluate_every = 5000
	steps_since_eval = 0

	current_state = env.reset()
	while len(actor_replay_buffer.buffer) < num_steps:
		print("\nCurrent step: {}".format(len(actor_replay_buffer.buffer)))

		# Sample from policy; maybe we don't reset the environment -> since this may bias the policy toward initial observations
		for j in range(trajectory_length):
			action = td3_policy.select_action(np.array(current_state))
			next_state, reward, done, _ = env.step(action)

			if done:
				actor_replay_buffer.addAbsorbing()
				current_state = env.reset()
			else:
				actor_replay_buffer.add((current_state, action, next_state), done)
				current_state = next_state

		discriminator.train(actor_replay_buffer, expert_buffer, trajectory_length, sum_or_mean_loss, batch_size)

		td3_policy.train(discriminator, actor_replay_buffer, trajectory_length, batch_size)

		if steps_since_eval >= evaluate_every:
			steps_since_eval = 0

			evaluation = (evaluate_policy(cl_args.env_id, td3_policy, 0), len(actor_replay_buffer))
			evaluations.append(evaluation)

		steps_since_eval += trajectory_length

	evaluations.append(evaluate_policy(cl_args.env_id, td3_policy))
	df = pd.DataFrame.from_records(evaluations)
	df.columns = ['rewards', 'timestep']
	df.to_csv('results/{}_{}_tsteps_results'.format(args.env_id, len(actor_replay_buffer)))


# Runs policy for X episodes and returns average reward
def evaluate_policy(env_name, policy, evaluation_trajectories=3):
	"""

	Args:
		env_name (str):
		policy:	The policy being evaluated
		evaluation_trajectories:

	Returns:

	"""
	env = gym.make(env_name)
	rewards = []
	for _ in range(evaluation_trajectories):
		r = 0.
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			r += reward
		rewards.append(r)

	# avg_reward = np.mean(rewards)
	# std_dev = np.std(rewards)

	# print ("---------------------------------------")
	# print ("Evaluation over %d episodes: %f" % (evaluation_trajectories, avg_reward))
	# print ("---------------------------------------")
	# with open("results/results.csv", "a") as f:
	# 	f.write(str(timestep) + "," + str(avg_reward) + "," + str(std_dev) + "\n")

	return rewards


if __name__ == '__main__':
	args = argsparser()
	main(args)
