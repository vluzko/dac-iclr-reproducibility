import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path

results_dir = Path(__file__).parent


def main():
	environments = ('Ant', 'Cheetah', 'Hopper', 'Walker')
	aggs = ('mean', 'sum')
	new_fname_template = "DAC_{}_{}_formatted.csv"
	plot_template = "DAC_{}_no_resets.png"
	for env in environments:
		mean_agg = pd.read_csv(str(results_dir / new_fname_template.format(env, 'mean')), index_col=0)
		x = mean_agg['timestep']
		reward = mean_agg['mean_reward']
		stdev = mean_agg['reward_stdev']
		upper = reward + stdev
		lower = reward - stdev

		fig, ax = plt.subplots()

		ax.plot(x, reward, color='red', label='Mean Loss')
		# ax.plot(x, upper)
		# ax.plot(x, lower)

		sum_agg = pd.read_csv(str(results_dir / new_fname_template.format(env, 'mean')), index_col=0)
		x = sum_agg['timestep']
		reward = sum_agg['mean_reward']
		stdev = sum_agg['reward_stdev']
		upper = reward + stdev
		lower = reward - stdev

		ax.plot(x, reward, color='blue', label='Mean Loss')
		# ax.plot(x, upper)
		# ax.plot(x, lower)

		plt.show()


if __name__ == '__main__':
	main()
