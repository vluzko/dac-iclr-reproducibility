# Discriminator-Actor-Critic
Note that all commands must be run from with the `dac` folder.
## Setup
1. Setup a Python 2 environment. (Commands for virtualenv given below)

	```
    pip install virtualenv
    virtualenv . -p $PYTHON2_PATH
    source bin/activate
	```
Note: You can use the command `which python2` to find the PYTHON2_PATH.

2. Installation (NOTE: need mujoco 1.31 setup -> download the zip from https://www.roboti.us/ and put the unzipped mjpro131 folder in the ~/.mujoco folder used for the mujoco license for gym-v1):

	```
    pip install -r requirements.txt
    ```

## Generate expert trajectories
DAC requires the expert trajectories to already exist. We use the OpenAI imitation repo to generate trajectories. (This is the original GAIL repo).

    git clone https://github.com/openai/imitation
    cd imitation
    python -m scripts/im_pipeline pipelines/im_pipeline.yaml 0_sampletrajs
    cp imitation_runs/modern_stochastic/trajs/* ../trajs
## Run DAC

    python dac.py --env_id=$environment_name --expert_path=$path/to/expert/traj
Example (Ant-v1):

    python dac.py --env_id=Ant-v1 --expert_path=trajs/trajs_ant.h5


The environment name can be `Hopper-v1`, `HalfCheetah-v1`, `Ant-v1`, or `Walker2d-v1`

The path to the expert trajectory should be a path to the correspond file in the `trajs` folder.


