import pandas as pd
import fire

from pathlib import Path
from matplotlib import pyplot as plt

plots_dir = Path(__file__).parent
environments = ('Hopper-v1', 'HalfCheetah-v1', 'Ant-v1', 'Walker2d-v1')
aggs = ('mean', 'sum')


def figure_3():
    expert = pd.read_csv(str(plots_dir / 'expert.csv'), index_col=0)

    for env in environments:
        random = pd.read_csv(str(plots_dir / 'Random_{}.csv'.format(env)), index_col=0)

        expert_score = expert.loc[env][0]

        fig, ax = plt.subplots()
        ax.set_title('{} Results'.format(env))
        ax.set_ylabel('Normalized Learner Score')
        ax.set_xlabel('Training Step')
        plt.ylim(-0.2, 1.3)

        for agg in aggs:
            results = pd.read_csv(str(plots_dir / 'DAC_{}_{}.csv'.format(env, agg)), index_col=0)
            timesteps = results['timestep']
            avg_reward = results.iloc[:, :-1].mean(axis=1)

            average_random_reward = random.iloc[:, :-1].mean(axis=1)
            average_random_reward.index = random['timestep']

            translated_rewards = avg_reward.values - average_random_reward.loc[timesteps].values
            normalized_rewards = translated_rewards / expert_score
            ax.plot(timesteps, normalized_rewards, label='Batch {}'.format(agg.capitalize()))

        ax.legend()
        plt.savefig(str(plots_dir / '{}_results.png'.format(env)))


if __name__ == '__main__':
    fire.Fire()
