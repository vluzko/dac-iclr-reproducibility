'''
We modified the mujoco_dset.py file from openAI/baselines/gail/dataset and used the load_dataset() function
from openAI/imitation/scripts/imitate_mj.py
'''


import h5py
import numpy as np

def load_dataset(filename, limit_trajs, data_subsamp_freq=1):
    # Load expert data
    with h5py.File(filename, 'r') as f:
        # Read data as written by vis_mj.py
        full_dset_size = f['obs_B_T_Do'].shape[0] # full dataset size
        dset_size = min(full_dset_size, limit_trajs) if limit_trajs is not None else full_dset_size

        exobs_B_T_Do = f['obs_B_T_Do'][:dset_size,...][...]
        exa_B_T_Da = f['a_B_T_Da'][:dset_size,...][...]
        exr_B_T = f['r_B_T'][:dset_size,...][...]
        exlen_B = f['len_B'][:dset_size,...][...]

    print ('Expert dataset size: {} transitions ({} trajectories)'.format(exlen_B.sum(), len(exlen_B)))
    print ('Expert average return:', exr_B_T.sum(axis=1).mean())

    return exobs_B_T_Do, exa_B_T_Da, exr_B_T


class Dset(object):
    def __init__(self, inputs, labels, weights, num_traj):
        self.inputs = inputs
        self.labels = labels
        self.num_traj = num_traj
        assert len(self.inputs) == len(self.labels)
        assert len(self.inputs) == len(weights)
        assert self.num_traj > 0

        #Calc probabilities from weights

        #If we have n samples (original states and newly added absorbing states)
        #and t trajectories -> t absorbing states

        # Thus, let p be probability of original sample
        # p*(n-t) + p*t/n = 1

        n = len(weights)
        t = self.num_traj

        p = 1.0/(n-t + t/n)
        self.probabilities = [p * weight for weight in weights]
        self.indicies = np.arange(len(inputs))

    def get_next_batch(self, batch_size):
        assert batch_size <= len(self.inputs)
        indicies = np.random.choice(self.indicies, batch_size, p=self.probabilities, replace=True)

        return self.inputs[indicies, :], self.labels[indicies, :]

# This takes in 1 trajectory's observations and actions
# and adds absorbing state 0-vector with same dimension as observation_space
# and add absorbing action 0-vector with same dimension as action_space
def WrapAbsorbingState(env, obs, acs):
    obs_space = env.observation_space
    acs_space = env.action_space

    # Use zero matrix as generic absorbing state
    absorbing_state = np.zeros((1,obs_space.shape[0]),dtype=np.float32)
    random_action = np.zeros_like(acs_space.sample(),dtype=np.float32).reshape(1, acs_space.shape[0])

    # At terminal state obs[-1], under action acs[-1] we go to absorbing state.
    new_obs = np.concatenate((obs, absorbing_state))
    new_acs = np.concatenate((acs, random_action))

    return new_obs, new_acs

class Mujoco_Dset(object):
    def __init__(self, env, expert_path, traj_limitation=-1):
        temp_obs, temp_acs, rets = load_dataset(expert_path, traj_limitation)

        obs = np.zeros((temp_obs.shape[0], temp_obs.shape[1] + 1, temp_obs.shape[2]))
        acs = np.zeros((temp_acs.shape[0], temp_acs.shape[1] + 1, temp_acs.shape[2]))
        n = traj_limitation * (temp_obs.shape[1] + 1)
        weights = [1 for i in range(n)]

        # iterate through trajectories and add absorbing state
        # Update importance weights for absorbing states
        for i in range(temp_obs.shape[0]):
            obs[i], acs[i] = WrapAbsorbingState(env, temp_obs[i], temp_acs[i])
            weights[(i+1)*len(obs[i]) - 1] = 1/n

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
        self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])

        self.rets = rets.sum(axis=1)
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = len(rets)
        self.num_transition = len(self.obs)
        self.dset = Dset(self.obs, self.acs, weights, traj_limitation)
        self.log_info()

    def log_info(self):
        print("Total trajectorues: %d" % self.num_traj)
        print("Total transitions: %d" % self.num_transition)
        print("Average returns: %f" % self.avg_ret)
        print("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size):
        return self.dset.get_next_batch(batch_size)

