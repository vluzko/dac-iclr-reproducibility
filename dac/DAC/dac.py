from DAC.util.learning_rate import LearningRate
from DAC.util.replay_buffer import ReplayBuffer
from networks.adversary import Discriminator
from networks.TD3 import TD3
from dataset.mujoco_dset import Mujoco_Dset
from DAC import config

import gym
import argparse
import numpy as np
import pandas as pd
import torch
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def argsparser():
	parser = argparse.ArgumentParser("DAC")
	parser.add_argument('--env_id', help='environment ID', default='Hopper-v1')
	parser.add_argument('--seed', help='RNG seed', type=int, default=0)
	parser.add_argument('--expert_path', type=str, default='trajs/trajs_hopper.h5')
	parser.add_argument('--loss', type=str, default='sum')
	parser.add_argument('--traj_num', help='Number of Traj', type=int, default=4)
	parser.add_argument("--loss_fn", help="Which loss function to use.", type=str, default="cross_entropy")
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
	discriminator = Discriminator(state_dim + action_dim, aggregate=cl_args.loss, loss=cl_args.loss_fn).to(device)

	# For storing temporary evaluations
	evaluations = [
		evaluate_policy(env, td3_policy, 0)
	]

	evaluate_every = 5000
	steps_since_eval = 0

	while len(actor_replay_buffer) < num_steps:
		print("\nCurrent step: {}".format(len(actor_replay_buffer.buffer)))
		current_state = env.reset()
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

		discriminator.learn(actor_replay_buffer, expert_buffer, trajectory_length, batch_size)

		td3_policy.train(discriminator, actor_replay_buffer, trajectory_length, batch_size)

		if steps_since_eval >= evaluate_every:
			steps_since_eval = 0

			evaluation = evaluate_policy(env, td3_policy, len(actor_replay_buffer))
			evaluations.append(evaluation)

		steps_since_eval += trajectory_length

	last_evaluation = evaluate_policy(env, td3_policy, len(actor_replay_buffer))
	evaluations.append(last_evaluation)

	store_results(evaluations, len(actor_replay_buffer), cl_args.loss, cl_args.loss_fn)


def store_results(evaluations, number_of_timesteps, loss_aggregate, loss_function):
	"""Store the results of a run.

	Args:
		evaluations:
		number_of_timesteps (int):
		loss_aggregate (str): The name of the loss aggregation used. (sum or mean)
		loss_function (str): The name of the loss function used.

	Returns:
		None
	"""

	df = pd.DataFrame.from_records(evaluations)
	number_of_trajectories = len(evaluations[0]) - 1
	columns = ["reward_{}".format(i) for i in range(number_of_trajectories)]
	columns.append("timestep")
	df.columns = columns

	timestamp = time.time()
	results_fname = 'DAC_{}_{}_tsteps_{}_loss_{}_agg_{}_results.csv'.format(args.env_id, number_of_timesteps, loss_function, loss_aggregate, timestamp)
	df.to_csv(str(config.results_dir / results_fname))


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, time_step, evaluation_trajectories=3):
	"""

	Args:
		env: The environment being trained on.
		policy:	The policy being evaluated
		time_step (int): The number of time steps the policy has been trained for.
		evaluation_trajectories (int): The number of trajectories on which to evaluate.

	Returns:
		(list)	- The time_step, followed by all the rewards.
	"""
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

	rewards.append(time_step)
	return rewards


if __name__ == '__main__':
	args = argsparser()
	main(args)
