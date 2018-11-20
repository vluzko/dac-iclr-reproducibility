# Singleton class to set learning rate + decay for all networks

'''
The paper states:
"We trained all networks with the Adam optimizer (Kingma & Ba,2014) and
decay learning rate by starting with initial learning rate of 10^−3 and decaying it by 0.5
every 10^5 training steps for the actor network."
'''

class LearningRate:
	__instance = None

	def __init__(self):
		if LearningRate.__instance != None:
			raise Exception("Singleton instantiation called twice")
		else:
			LearningRate.__instance = self
			self.lr = None
			self.decayFactor = None

	@staticmethod
	def getInstance():
		if LearningRate.__instance == None:
			LearningRate()
		return LearningRate.__instance

	def setLR(self, lr):
		self.lr = lr

	def getLR(self):
		return self.lr

	def setDecay(self, d):
		self.decayFactor = d

	def decay(self):
		self.lr = self.lr * self.decayFactor