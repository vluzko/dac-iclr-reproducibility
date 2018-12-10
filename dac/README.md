# Discriminator-Actor-Critic

1. Setup virtualenv (WITH Python2 - you can use the command *which python2* to find the PYTHON2 PATH)
	```
    pip install virtualenv
	virtualenv . -p {PYTHON2 PATH}
	source bin/activate
	```

2. Installation (NOTE: need mujoco 1.31 setup -> download the zip from https://www.roboti.us/ and put the unzipped mjpro131 folder in the ~/.mujoco folder used for the mujoco license for gym-v1):
	```
    pip install -r requirements.txt
    cd dac
    pip install -e .
    ```

3. Generate Trajectory Information (for Hopper, walker, ant, half-cheetah):
    We'll use the original GAIL implementation to generate trajectories.

    ```
    git clone https://github.com/openai/imitation
    cd imitation
    python -m scripts/im_pipeline pipelines/im_pipeline.yaml 0_sampletrajs
    cp imitation_runs/modern_stochastic/trajs/* ../DAC/trajs
    ```
4. Run DAC

    ```
    python dac.py
    ```

