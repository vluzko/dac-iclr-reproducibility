class LearningRate:
	__instance = None

	def __init__(self):
		if LearningRate.__instance != None:
			raise Exception("Singleton instantiation called twice")
		else:
			LearningRate.__instance = self
			self.lr = None
			self.decayFactor = None
			self.training_step = 0

	@staticmethod
	def getInstance():
		if LearningRate.__instance == None:
			LearningRate()
		return LearningRate.__instance

	def setLR(self, lr):
		self.lr = lr

	def getLR(self):
		return self.lr

	def incrementStep(self):
		self.training_step += 1

	def getStep(self):
		return self.training_step

	def setDecay(self, d):
		self.decayFactor = d

	def decay(self):
		self.lr = self.lr * self.decayFactor