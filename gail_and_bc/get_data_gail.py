# python -m baselines.gail.run_getDataGAIL

from os import listdir
from os.path import isfile, join
import re
import os

checkpointGail = 'checkpoint/trpo_gail.transition_limitation_4.Hopper.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0/'
gailPrefix = 'trpo_gail.transition_limitation_4.Hopper.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0'

onlyfiles = [f for f in listdir(checkpointGail) if isfile(join(checkpointGail, f))]
gailFiles = [f for f in onlyfiles if re.match(r".*seed_0(\d)+", f)]

timesteps = []
for f in gailFiles:
    timesteps.append(int(f.split("seed_0")[1]))

timesteps.sort()

for i in timesteps:
    os.system('python get_data.py --task=evaluate --data_timestep=' + str(
        i) + ' --load_model_path=' + checkpointGail + gailPrefix + str(i))
