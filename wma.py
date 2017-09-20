import worlds
# import experts
import numpy as np
from matplotlib import pyplot as plt
import math 

# Bruce Willis
def expert_1(time_step):
	return 1

# glass is half empty
def expert_2(time_step):
	return -1

# odd-even rule 
def expert_3(time_step):
	if not time_step%2:
		return 1 # win if even
	else:
		return -1

def weighted_majority(expert_pred, expert_weights):
	return np.sign(np.dot(expert_pred, expert_weights))

def sample_rwma(weights):
	sum_weights = 0
	weight_probability = [0]*3
	sum_probability = 0
	sum_weights = math.fsum(weights)
	for i in range(3):
		weight_probability[i] = 1.0*weights[i]/sum_weights
		sum_probability = sum_probability + weight_probability[i]
		weight_probability[i] = sum_probability

	for i in range(3):
		if random.random() < weight_probability[i]:
			return i

def randomized_weighted_majority(expert_pred, expert_weights):
	y_pred = sample_rwma(w,h)

def calculate_loss(pred, label):
	loss = 0
	if pred==label:
		loss = 1
	return loss

def q_3_3(world_type, eta, n_time_steps):
	world = worlds.World(world_type=world_type)
	weights = np.ones(3)
	loss_experts = np.zeros(3)
	# list of lists
	cumulative_loss_learner = np.zeros(n_time_steps)
	sum_loss_experts = np.zeros(3)
	cumulative_loss_experts = [[0 for i in range (3)] for j in range(n_time_steps)]
	sum_loss_learner = 0

	for time_step in range(n_time_steps):
		expert_pred = [expert_1(time_step), \
						expert_2(time_step), \
						expert_3(time_step)]
		y_pred = weighted_majority(expert_pred, weights)
		# check implementation in worlds.py
		y_label = world.get_label(time_step=time_step, 
									expert_pred=expert_pred, 
									expert_weights=weights)
		# update weights
		# todo vectorize

		loss_learner = calculate_loss(y_pred, y_label)
		sum_loss_learner = sum_loss_learner + loss_learner
		cumulative_loss_learner[time_step] = sum_loss_learner

		# loss_experts = [calculate_loss(pred, y_label) for pred in expert_pred]

		for i in range(3):
			weights[i] = weights[i]*(1-(eta*(expert_pred[i]!=y_label)))
			loss_experts[i] = calculate_loss(expert_pred[i], y_label)
			sum_loss_experts[i] = sum_loss_experts[i] + loss_experts[i]
			cumulative_loss_experts[time_step][i] = sum_loss_experts[i]

	plot_loss(cumulative_loss_learner, cumulative_loss_experts)
	plot_regret(cumulative_loss_learner, cumulative_loss_experts, n_time_steps)
	plt.show()

def q_3_4(world_type, eta, n_time_steps):
	world = worlds.World(world_type=world_type)
	weights = np.ones(3)
	loss_experts = np.zeros(3)
	# list of lists
	cumulative_loss_learner = np.zeros(n_time_steps)
	sum_loss_experts = np.zeros(3)
	cumulative_loss_experts = [[0 for i in range (3)] for j in range(n_time_steps)]
	sum_loss_learner = 0

	for time_step in range(n_time_steps):
		expert_pred = [expert_1(time_step), \
						expert_2(time_step), \
						expert_3(time_step)]
		y_pred = weighted_majority(expert_pred, weights)
		# check implementation in worlds.py
		y_label = world.get_label(time_step=time_step, 
									expert_pred=expert_pred, 
									expert_weights=weights)
		# update weights
		# todo vectorize

		loss_learner = calculate_loss(y_pred, y_label)
		sum_loss_learner = sum_loss_learner + loss_learner
		cumulative_loss_learner[time_step] = sum_loss_learner

		# loss_experts = [calculate_loss(pred, y_label) for pred in expert_pred]

		for i in range(3):
			weights[i] = weights[i]*(1-(eta*(expert_pred[i]!=y_label)))
			loss_experts[i] = calculate_loss(expert_pred[i], y_label)
			sum_loss_experts[i] = sum_loss_experts[i] + loss_experts[i]
			cumulative_loss_experts[time_step][i] = sum_loss_experts[i]

	plot_loss(cumulative_loss_learner, cumulative_loss_experts)
	plot_regret(cumulative_loss_learner, cumulative_loss_experts, n_time_steps)
	plt.show()


def plot_loss(cumulative_loss_learner, cumulative_loss_experts):
	plt.figure()
	expert_1, = plt.plot([row[0] for row in cumulative_loss_experts],'r-',label='expert_1')
	expert_2, = plt.plot([row[1] for row in cumulative_loss_experts],'b-',label = 'expert_2')
	expert_3, = plt.plot([row[2] for row in cumulative_loss_experts],'y-',label = 'expert_3')
	learner, = plt.plot(cumulative_loss_learner,'g-',label = 'learner')
	plt.legend([expert_1, expert_2, expert_3, learner],['expert_1','expert_2','expert_3','learner'])
	plt.xlabel('time')
	plt.ylabel('Cumulative Loss')

def plot_regret(cumulative_loss_learner,cumulative_loss_experts,n_time_steps):
	plt.figure()
	loss_best_expert = [0]*n_time_steps
	regret = [0]*n_time_steps
	sum_regret = 0
	avg_regret = [0]*n_time_steps
	for time_step in range(n_time_steps):
		loss_best_expert[time_step] = min(cumulative_loss_experts[time_step])
		regret[time_step] = cumulative_loss_learner[time_step] - loss_best_expert[time_step]
		avg_regret[time_step] = 1.0*regret[time_step]/(time_step+1)

	plt.plot(avg_regret)
	plt.xlabel('time')
	plt.ylabel('Average Cumulative Regret')
	# plt.title('Deterministic with extra observation and expert')

if __name__ == '__main__':
	q_3_3(world_type="stochastic", eta=0.5, n_time_steps=100)
	q_3_3(world_type="deterministic", eta=0.5, n_time_steps=100)
	q_3_3(world_type="adversarial", eta=0.5, n_time_steps=100)
	q_3_4(world_type="stochastic", eta=0.5, n_time_steps=100)
	q_3_4(world_type="deterministic", eta=0.5, n_time_steps=100)
	q_3_4(world_type="adversarial", eta=0.5, n_time_steps=100)
