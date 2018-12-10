# Baselines
This directory contains code for running behavioral cloning and GAIL. The implementations of both algorithms are from the OpenAI baselines repository.

## Setup


1. Setup virtualenv (WITH Python3)
    ```
    pip install virtualenv
    virtualenv . -p {PYTHON3 PATH}
    source bin/activate
    ```

2. Installation (NOTE: need mujoco 1.50 setup for baselines installation):
    ```
    pip install -r requirements.txt
    ```

3. Get the expert data
    * Download the expert data from https://drive.google.com/drive/folders/1h3H4AY_ZBx08hz-Ct0Nxxus-V1melu1U
    * Put it in gail_and_bc/data


## Run GAIL
Note that the following commands must be run from within `gail_and_bc`.
    ```
    python run_mujoco.py --env_id=$environment --expert_path=data/$expert_data.npz --traj_limitation=4 --save_per_iter=5 --num_timesteps=1100000
    ```

Example (Hopper-v2, 4 expert trajectories, saves every 5 iterations, runs for 1,100,000 timesteps):
    ```
    python run_mujoco.py --env_id=Hopper-v2 --expert_path=data/deterministic.trpo.Hopper.0.00.npz --traj_limitation=4 --save_per_iter=5 --num_timesteps=1100000
    ```

## Evaluating GAIL
    ```
    python get_data_gail.py
    ```

This will evaluate for every timestep saved in checkpoint/{GAIL_DIR}/. Running this code will generate the file `gail.csv` containing all of the results for GAIL.

## Run Behavioural Cloning
    ```
    python behavior_clone.py --env_id=Hopper-v2 --expert_path=data/deterministic.trpo.Hopper.0.00.npz --traj_limitation=4 --BC_max_iter=1100
    ```
    
**NOTE: Behavioural clone fails after training, but the results are still saved. The failure occurs during the evaluation of the training**

## Evaluating BC
    ```
    python get_data_bc.py
    ```

This stores results in `BC.csv`.


## Plotting
    ```
    cd plot
    jupyter notebook
    Open plot.ipynb -> change the .csv file paths to the newly generated results and run
    ```
