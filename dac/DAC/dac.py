from util.learning_rate import LearningRate
from util.replay_buffer import ReplayBuffer
from networks.adversary import Discriminator
from networks.TD3 import TD3
from dataset.mujoco_dset import Mujoco_Dset
import gym
import argparse
import numpy as np
import torch

def argsparser():
    parser = argparse.ArgumentParser("DAC")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='trajs/trajs_hopper.h5')
    parser.add_argument('--traj_num', help='Number of Traj', type=int, default=4)
    return parser.parse_args()



def main(args):
	env = gym.make(args.env_id)
	expert_buffer = Mujoco_Dset(env, args.expert_path, args.traj_num) # The buffer for the expert -> refer to dataset/mujoco_dset.py
	actor_replay_buffer = ReplayBuffer(env)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	lr = LearningRate.getInstance()
	lr.setLR(10**(-3)) # Loss is 10e-3
	lr.setDecay(1.0/2.0) # Decay is 1/2

	# TD3(state_dim, action_dim, max_action, actor_clipping, decay_steps*) *Not used yet;
	td3 = TD3(state_dim, action_dim, max_action, 40, 10**5)

	# Input dim = state_dim + action_dim
	discriminator = Discriminator(state_dim + action_dim)
	if torch.cuda.is_available(): discriminator = discriminator.to('cuda')

	batch_size = 100 #openReview: they state they do batch_size of 100
	num_steps = 1e6 # 1 million timesteps -> in paper, they go to 1 million timesteps
	T = 1000 # Trajectory length == T in the pseudo-code; 1000 is stated in openReview

	evaluate_policy(args.env_id, td3, 0)

	evaluate_every = 5000
	steps_since_eval = 0

	obs = env.reset()
	while len(actor_replay_buffer.buffer) < num_steps: # Do 1 million timesteps from the environment
		# Sample from policy; maybe we don't reset the environment -> since this may bias the policy toward initial observations
		# obs = env.reset() -> I'm going to make this reset change now
		for j in range(T):
			action = td3.select_action(np.array(obs))
			next_state, reward, done, _ = env.step(action)
			actor_replay_buffer.add((obs, action, next_state), done)
			steps_since_eval += 1
			if done:
				actor_replay_buffer.addAbsorbing()
				obs = env.reset()
			else:
				obs = next_state

		discriminator.train(actor_replay_buffer, expert_buffer, T, batch_size)

		td3.train(discriminator, actor_replay_buffer, T, batch_size) #discriminator, replay_buf, iterations, batch_size=100

		if steps_since_eval >= evaluate_every:
			steps_since_eval = steps_since_eval % evaluate_every
			evaluate_policy(args.env_id, td3, len(actor_replay_buffer.buffer))
	evaluate_policy(args.env_id, td3, len(actor_replay_buffer.buffer))

# Runs policy for X episodes and returns average reward
def evaluate_policy(env_name, policy, timestep, eval_episodes=10):
	env = gym.make(env_name)
	rewards = []
	for _ in range(eval_episodes):
		r = 0.
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			r += reward
		rewards.append(r)

	avg_reward = np.mean(rewards)
	std_dev = np.std(rewards)

	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print ("---------------------------------------")
	with open("results.csv", "a") as f:
		f.write(str(timestep) + "," + str(avg_reward) + "," + str(std_dev) + "\n")


if __name__ == '__main__':
	args = argsparser()
	main(args)


