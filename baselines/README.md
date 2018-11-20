1. Setup virtualenv (WITH Python3)
	```
    pip install virtualenv
	virtualenv .
	source bin/activate
	```

2. Installation (NOTE: need mujoco 1.50 setup for baselines installation):
	```
    pip install -r requirements.txt
    ```

3. Getting data:
    Refer to **baselines/data/README.md**

4. For the following commands, make sure you are in baselines folder:
	```
	cd baselines
    ```
    
5. Running GAIL:
    _HopperV2, 4 expert trajectories, save every 5 iterations, for 1,100,000 timesteps_
    ```	
    python run_mujoco.py --env_id=Hopper-v2 --expert_path=data/deterministic.trpo.Hopper.0.00.npz --traj_limitation=4 --save_per_iter=5 --num_timesteps=1100000
    ```

**NOTE: 1 iteration involves many timesteps**

6. Evaluating GAIL:
    ```
	python run_getDataGAIL.py
	```

**NOTE: This will evaluate for every timestep saved in checkpoint/{GAIL_DIR}/. Running this code will generate gail.csv**

7. Running Behavioural Clone
    ```	
    python behavior_clone.py --env_id=Hopper-v2 --expert_path=data/deterministic.trpo.Hopper.0.00.npz --traj_limitation=4 --BC_max_iter=1100
    ```
    
**NOTE: Change behavior_clone.py line 71 if you want to save at different iterations. Currently, saves every 50 iterations. Also, max iterations = 1100, since the batch size is 1024. Does this mean 1100 x 1024 timesteps?**

**NOTE: Behavioural clone fails after training, but the results are still saved. The failure occurs during the evaluation of the training**

8. Evaluating BC:
	```
    python run_getDataBC.py
    ```

**NOTE: This generates BC.csv**


9. Plotting:
	```
    cd plot
	jupyter notebook
	Open plot.ipynb -> change the .csv file paths to the newly generated results and run
    ```
