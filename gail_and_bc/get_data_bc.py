# python -m baselines.gail.run_getDataGAIL

from os import listdir
from os.path import isfile, join
import re
import os

checkpointBC = 'checkpoint/BC.Hopper.traj_limitation_4.seed_0/'
bcPrefix = 'BC.Hopper.traj_limitation_4.seed_0'

onlyfiles = [f for f in listdir(checkpointBC) if isfile(join(checkpointBC, f))]
bcFiles = [f for f in onlyfiles if re.match(r".*seed_0(\d)+", f)]

timesteps = []
for f in bcFiles:
    timesteps.append(int(f.split("seed_0")[1]))

timesteps.sort()

for i in timesteps:
    os.system('python get_data.py --outputFile=BC --task=evaluate --data_timestep=' + str(
        i) + ' --load_model_path=' + checkpointBC + bcPrefix + str(i))
