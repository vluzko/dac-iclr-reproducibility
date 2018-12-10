class LearningRate:
    """
    Attributes:
        lr (float)
        decay_factor (float)
        training_step (int)
    """
    __instance = None

    def __init__(self):
        if LearningRate.__instance is not None:
            raise Exception("Singleton instantiation called twice")
        else:
            LearningRate.__instance = self
            self.lr = None
            self.decay_factor = None
            self.training_step = 0

    @staticmethod
    def get_instance():
        """Get the singleton instance.

        Returns:
            (LearningRate)
        """
        if LearningRate.__instance is None:
            LearningRate()
        return LearningRate.__instance

    def set_learning_rate(self, lr):
        self.lr = lr

    def get_learning_rate(self):
        return self.lr

    def increment_step(self):
        self.training_step += 1

    def get_step(self):
        return self.training_step

    def set_decay(self, d):
        self.decay_factor = d

    def decay(self):
        self.lr = self.lr * self.decay_factor
