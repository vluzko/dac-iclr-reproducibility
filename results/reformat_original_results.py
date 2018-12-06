"""Reformat the original results format to something more usable"""
import pandas as pd

from pathlib import Path
from itertools import product

results_dir = Path(__file__).parent


def load_and_reformat(fname: str) -> pd.DataFrame:
	df = pd.read_csv(fname)
	df.columns = ['timestep', 'mean_reward', 'reward_stdev']
	return df


def main():
	environments = ('Ant', 'Cheetah', 'Hopper', 'Walker')
	aggs = ('mean', 'sum')
	template = "DAC_{}_cross_entropy_loss_{}_agg.csv"
	new_fname_template = "DAC_{}_{}_formatted.csv"

	for env, agg in product(environments, aggs):
		fname = results_dir / template.format(env, agg)
		reformated = load_and_reformat(str(fname))
		new_fname = results_dir / new_fname_template.format(env, agg)
		reformated.to_csv(str(new_fname))


if __name__ == '__main__':
	main()
