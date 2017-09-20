import random
import numpy as np

class World:
	def __init__(self, world_type):
		# string: "stochastic" /  "deterministic" / "adversarial"
		self.world_type = world_type 
		self.labels = [-1, 1] # lost or won
		self.win_every_nth = 3 # sets our deterministic world

	def get_label(self, **kwargs):
		if self.world_type == "stochastic":
			return random.choice(self.labels)

		# tartan's wins if kwargs['time_step'] is divisible by self.win_every_nth
		if self.world_type == "deterministic":
			if not kwargs['time_step']%self.win_every_nth:
				return 1
			else:
				return -1 

		# listens to kwargs['expert_pred'], and kwargs['expert_weights']
		if self.world_type == "adversarial":
			# declutter your code with this one simple trick 
			expert_pred = kwargs['expert_pred'] # type: list
			expert_weights = kwargs['expert_weights'] # type: list
			
			# calculate weighted sum
			wma_pred = np.sign(np.dot(expert_pred, expert_weights)) # -1 if x < 0, 0 if x==0, 1 if x > 0

			# hashtag be the adversary you want to see in the world. 
			if wma_pred == 0 or wma_pred == 1:
				return -1
			else:
				return 1