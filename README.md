# metalearning

## setup
1. load new software stack
    `env2lmod`
2. load modules
    ```
    module load gcc/6.3.0 python_gpu/3.7.4 tmux/2.6 eth_proxy
    module load mesa-glu/9.0.0                                  # is needed for mujoco
    module load glfw                                            # is needed for mujoco
    module load bazel                                           # needed for dm-tree if python 3.7 (garage)
    ```
    All in one line
    ```
    module load gcc/6.3.0 python_gpu/3.7.4 tmux/2.6 eth_proxy mesa-glu/9.0.0 glfw bazel
    ```
3. Install mujoco_py dependencies
    ```
    sh mujoco.sh
    source ~/.bashrc
    ```
```
python -m venv rl
source ./rl/bin/activate
pip install -r ./requirements.txt
```

```
bsub -n 12 -J "maml-tpro" -W 4:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python maml_trpo_metaworld_ml10.py'
```

```
bsub -n 8 -J "maml-tpro" -W 4:00 -R "rusage[mem=10240, ngpus_excl_p=1]" 'python maml_trpo_metaworld_ml10.py'
```
```
bjobs -w
bbjobs
bpeek -f
```
