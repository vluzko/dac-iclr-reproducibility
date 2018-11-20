# Discriminator-Actor-Critic

1. Setup virtualenv (WITH Python2)
	```
    pip install virtualenv
	virtualenv . -p {PYTHON2 PATH}
	source bin/activate
	```

2. Installation (NOTE: need mujoco 1.31 setup for gym-v1):
	```
    pip install -r requirements.txt
    ```

3. Generate Trajectory Information (for Hopper, walker, ant, half-cheetah):
    We'll use the original GAIL implementation to generate trajectories

    ```
    git clone https://github.com/openai/imitation
    cd imitation
    python -m scripts/im_pipeline phase=0_sampletrajs spec=pipelines/im_pipeline.yaml
    cp imitation_runs/modern_stochastic/trajs/* ../DAC/trajs
    ```
4. Run DAC

