# Code based on:
# https://github.com/sfujim/TD3/blob/master/TD3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

from DAC.util import replay_buffer, learning_rate
# from util.learning_rate import LearningRate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)

		self.max_action = max_action

	def forward(self, x):
		if not torch.cuda.is_available(): x = x.float()
		x = torch.relu(self.l1(x))
		x = torch.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x))
		return x

	def act(self, x):
		x = torch.FloatTensor(x).to(device)
		return self(x)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)

	def forward(self, x, u):
		xu = torch.cat([x, u], 1)
		# if not torch.cuda.is_available(): xu = xu.float()

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)

		x2 = F.relu(self.l4(xu))
		x2 = F.relu(self.l5(x2))
		x2 = self.l6(x2)
		return x1, x2

	def Q1(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)
		return x1


class TD3(object):
	def __init__(self, state_dim, action_dim, max_action, actor_clipping, decay_steps):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.decay_steps = decay_steps
		self.actor_grad_clipping = actor_clipping
		self.max_action = max_action
		self.actor_steps = 0
		self.critic_steps = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def adjust_critic_learning_rate(self, lr):
		print("Setting critic learning rate to: {}".format(lr))
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

	def adjust_actor_learning_rate(self, lr):
		print("Setting actor learning rate to: {}".format(lr))
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

	def sample(self, state):
		next_state = self.actor(torch.FloatTensor(state).to(device))
		return torch.FloatTensor(next_state).to(device)

	def reward(self, discriminator, states, actions):
		states_actions = torch.cat([states, actions], 1).to(device)
		return discriminator.reward(states_actions)

	def train(self, discriminator, replay_buf, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
			  noise_clip=0.5, policy_freq=2):

		lr_tracker = learning_rate.LearningRate.get_instance()
		lr = lr_tracker.lr

		self.adjust_actor_learning_rate(lr)
		self.adjust_critic_learning_rate(lr)

		for iteration in range(iterations):
			# Sample replay buffer
			x, y, u, d = replay_buf.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(y).to(device)
			next_state = torch.FloatTensor(u).to(device)

			reward = self.reward(discriminator, state, action)
			# Select action according to policy and add clipped noise
			noise = torch.FloatTensor(y).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
			next_action = next_action.clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			# target_Q = reward + (done * discount * target_Q).detach()
			target_Q = reward + (discount * target_Q).detach()

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			if iteration == 0 or iteration == iterations - 1:
				print("Critic Iteration: {:3} ---- Loss: {:.5f}".format(iteration, critic_loss.item()))
			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if iteration % policy_freq == 0:

				# Compute actor loss
				actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
				if iteration == 0 or iteration == iterations - 1 or iteration == iterations - 2:
					print("Actor Iteration:  {:3} ---- Loss: {:.5f}".format(iteration, actor_loss.item()))
				# Optimize the actor
				self.actor_optimizer.zero_grad()
				actor_loss.backward()

				# Clip, like the paper
				clip_grad_value_(self.actor.parameters(), self.actor_grad_clipping)

				self.actor_optimizer.step()
				lr_tracker.training_step += 1
				step = lr_tracker.training_step

				if step != 0 and step % self.decay_steps == 0:
					print("Decaying learning rate at step: {}".format(step))
					lr_tracker.decay()

					self.adjust_actor_learning_rate(lr_tracker.lr)
					self.adjust_critic_learning_rate(lr_tracker.lr)

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
