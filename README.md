# Meta Learning

## Inital one time setup
1. Load the new software stack
    `env2lmod`
2. Load the software modules
    ```
    module load gcc/6.3.0 python_gpu/3.7.4 tmux/2.6 eth_proxy
    module load mesa-glu/9.0.0                                  # is needed for mujoco
    module load glfw/3.3.4                                      # is needed for mujoco
    module load bazel                                           # needed for dm-tree if python 3.7 (garage)
    ```
    all in one line
    ```
    module load gcc/6.3.0 python_gpu/3.7.4 mesa-glu/9.0.0 glfw/3.3.4 bazel/3.7.1 tmux/2.6 eth_proxy
    ```
3. Install the mujoco_py dependencies
    ```
    sh mujoco.sh
    source ~/.bashrc
    ```
4. Install the pyhton environment
    ```
    python -m venv rl
    source ./rl/bin/activate
    pip install -r ./requirements.txt
    ```

## Every time setup
```
env2lmod
module load gcc/6.3.0 python_gpu/3.7.4 mesa-glu/9.0.0 glfw/3.3.4 bazel/3.7.1 tmux/2.6 eth_proxy
cd metalearning
source ./rl/bin/activate
```


## Running a job
### cpu smaller job (10*3Gb)
```
bsub -n 10 -J "maml-tpro" -W 4:00 -R "rusage[mem=3072]" 'python maml_trpo_metaworld_ml10.py'
```

### gpu smaller job (10*3Gb & any gpu)
```
bsub -n 10 -J "maml-tpro" -W 4:00 -R "rusage[mem=3072, ngpus_excl_p=1]" 'python maml_trpo_metaworld_ml10.py'
```

### gpu larger job (12*10Gb & 2080Ti)
```
bsub -n 12 -J "maml-tpro" -W 4:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python maml_trpo_metaworld_ml10.py'
```

## Some useful cluster commands
### jobs
```
bbjobs
bjobs -w
bjobs -l
bpeek -f
```

### modules
```
module ls                      # list loaded modules
module spder python            # search for modules with name pyhton
```


## Troubleshooting
In case of the error
```
...
File "mujoco_py/cymj.pyx", line 1, in init mujoco_py.cymj
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
```
reinstall mujocopy with the numpy version of your liking. For instance numpy 1.19.15 is compatible with tensorflow (see https://github.com/openai/mujoco-py/issues/607).
```
pip cache remove mujoco_py
pip uninstall mujoco_py
# install numpy version you like to use before installing mujoco-py
pip install numpy==1.19.5 six~=1.15.0
pip install mujoco-py --no-cache-dir --no-binary :all: --no-build-isolation
```
