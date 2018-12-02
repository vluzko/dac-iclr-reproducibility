# Based on https://github.com/higgsfield/RL-Adventure-2/blob/master/8.gail.ipynb

import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util.learning_rate import LearningRate
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# entropy_weight = 0.001 from openAI/imiation
class Discriminator(nn.Module):
	def __init__(self, num_inputs, hidden_size=100, lamb=10, entropy_weight=0.001):
		super(Discriminator, self).__init__()

		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)
		self.linear3.weight.data.mul_(0.1)
		self.linear3.bias.data.mul_(0.0)
		self.criterion = nn.BCEWithLogitsLoss()
		self.entropy_weight = entropy_weight
		self.optimizer = torch.optim.Adam(self.parameters())
		self.LAMBDA = lamb # used in gradient penalty
		self.use_cuda = torch.cuda.is_available()

	def forward(self, x):
		if not self.use_cuda: x = x.float()
		x = torch.tanh(self.linear1(x))
		x = torch.tanh(self.linear2(x))
		# prob = torch.sigmoid(self.linear3(x))
		# return prob
		out = self.linear3(x)
		return out

	def reward(self, x):
		out = self(x)
		probs = torch.sigmoid(out)
		return torch.log(probs + 1e-8) - torch.log(1-probs + 1e-8)


	def adjust_adversary_learning_rate(self, lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def logit_bernoulli_entropy(self, logits):
		ent = (1. - torch.sigmoid(logits)) * logits - self.logsigmoid(logits)
		return ent

	def logsigmoid(self, a):
		return torch.log(torch.sigmoid(a))

	def logsigmoidminus(self, a):
		return torch.log(1-torch.sigmoid(a))

	def train(self, replay_buf, expert_buf, iterations, sum=True, batch_size=100):
		lr = LearningRate.get_instance().get_learning_rate()
		self.adjust_adversary_learning_rate(lr)

		for it in range(iterations):
			# Sample replay buffer
			x, y, u, d = replay_buf.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(y).to(device)
			next_state = torch.FloatTensor(u).to(device)

			# Sample expert buffer
			expert_obs, expert_act, expert_weights = expert_buf.get_next_batch(batch_size)
			expert_obs = torch.FloatTensor(expert_obs).to(device)
			expert_act = torch.FloatTensor(expert_act).to(device)
			expert_weights = torch.FloatTensor(expert_weights).to(device).view(-1, 1)

			# Predict
			state_action = torch.cat([state, action], 1).to(device)
			expert_state_action = torch.cat([expert_obs, expert_act], 1).to(device)

			fake = self(state_action)
			real = self(expert_state_action)

			gradient_penalty = self._gradient_penalty(expert_state_action, state_action)
			if sum:
				gen_loss = torch.sum(self.logsigmoidminus(fake)).to(device)
				expert_loss = torch.sum((self.logsigmoid(real) * expert_weights)).to(device)
			else:
				gen_loss = torch.mean(self.logsigmoidminus(fake)).to(device)
				expert_loss = torch.mean((self.logsigmoid(real) * expert_weights)).to(device)

			logits = torch.cat([fake,real], 0)
			entropy = torch.mean(self.logit_bernoulli_entropy(logits))
			entropy_loss = -self.entropy_weight * entropy

			self.optimizer.zero_grad()

			# I think the pseudo-code loss is wrong. Refer to equation (2) of paper :
			# MaxD E(D(s,a)) + E(1-D(s',a')) -> minimizing the negative of this
			total_loss = -(gen_loss + expert_loss) + entropy_loss + gradient_penalty

			if it == 0 or it == iterations - 1:
				print("Iteration: " + str(it) + " ---- Loss: " + str(total_loss) + " | Expert_loss: " + str(expert_loss) + " | Gen_loss: " + str(gen_loss) + " | Fake Prob: " + str(torch.sigmoid(fake[0])) + " | Real Prob: " + str(torch.sigmoid(real[0])))
			total_loss.backward()
			self.optimizer.step()

	# From https://github.com/EmilienDupont/wgan-gp/blob/master/training.py -> _gradient_penalty()
	def _gradient_penalty(self, real_data, generated_data):
		discriminator = self
		batch_size = real_data.size()[0]

		# Calculate interpolation
		alpha = torch.rand(batch_size, 1)
		alpha = alpha.expand_as(real_data)
		if self.use_cuda:
			alpha = alpha.cuda()
		interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
		interpolated = Variable(interpolated, requires_grad=True)
		if self.use_cuda:
			interpolated = interpolated.cuda()

		# Calculate probability of interpolated examples
		prob_interpolated = discriminator(interpolated)

		# Calculate gradients of probabilities with respect to examples
		gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
							   grad_outputs=torch.ones(
								   prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
								   prob_interpolated.size()),
							   create_graph=True, retain_graph=True)[0]

		# Gradients have shape (batch_size, num_channels, img_width, img_height),
		# so flatten to easily take norm per example in batch
		gradients = gradients.view(batch_size, -1)

		# Derivatives of the gradient close to 0 can cause problems because of
		# the square root, so manually calculate norm and add epsilon

		#gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

		# Return gradient penalty
		#return self.LAMBDA * ((gradients_norm - 1) ** 2).mean()
		return self.LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

