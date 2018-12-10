import argparse
import os.path as osp
import logging
import gym
import numpy as np

from os.path import isfile
from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger


from run_mujoco import runner


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--outputFile', type=str, default='gail')
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=5)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--data_timestep', type=int, default=-1)
    return parser.parse_args()


def get_task_name(args):
    task_name = args.algo + "_gail."
    if args.pretrained:
        task_name += "with_pretrained."
    if args.traj_limitation != np.inf:
        task_name += "transition_limitation_%d." % args.traj_limitation
    task_name += args.env_id.split("-")[0]
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
                ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    task_name += ".seed_" + str(args.seed)
    return task_name


args = argsparser()
U.make_session(num_cpu=1).__enter__()
set_global_seeds(args.seed)
env = gym.make(args.env_id)


def policy_fn(name, ob_space, ac_space, reuse=False):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)


env = bench.Monitor(env, logger.get_dir() and
                    osp.join(logger.get_dir(), "monitor.json"))
env.seed(args.seed)
gym.logger.setLevel(logging.WARN)
task_name = get_task_name(args)
args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
args.log_dir = osp.join(args.log_dir, task_name)

a, b, c = runner(env, policy_fn,
                 args.load_model_path,
                 timesteps_per_batch=1024,
                 number_trajs=10,
                 stochastic_policy=args.stochastic_policy,
                 save=args.save_sample
                 )

if isfile(args.outputFile + '.csv'):
    with open(args.outputFile + '.csv', "a") as f:
        f.write(str(args.data_timestep) + "," + str(a) + "," + str(b) + "," + str(c) + "\n")
else:
    with open(args.outputFile + '.csv', "w+") as f:
        f.write("Timestep, Average Length, Average Reward, Std\n")
        f.write(str(args.data_timestep) + "," + str(a) + "," + str(b) + "," + str(c) + "\n")
