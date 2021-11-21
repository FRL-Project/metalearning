# Meta Learning

## Inital one time setup
1. Load the new software stack
    `env2lmod`
2. Load the software modules
    ```
    module load gcc/6.3.0 python_gpu/3.7.4 tmux/2.6 eth_proxy
    # is needed for mujoco
    module load mesa-glu/9.0.0
    module load glfw/3.3.4
    # needed for dm-tree if python 3.7 (garage)
    module load bazel
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
5. Add .env file in root directory and paste following content:
    ```
    OUT_DIR=/cluster/scratch/<username>
    ```

## Every time setup
```
env2lmod
module load gcc/6.3.0 python_gpu/3.7.4 mesa-glu/9.0.0 glfw/3.3.4 bazel/3.7.1 tmux/2.6 eth_proxy
cd metalearning
source ./rl/bin/activate
```

## Running a job

List of specific commands for the experiments.
Commands are customized such that a job needs the right amout of resources (check with `bbjobs`), to get better priority; As well as fixing a gpu for reproducibility.

| experiment               | epoch time | cmd                                                                                                |
|--------------------------|------------|----------------------------------------------------------------------------------------------------|
| maml_trpo_metaworld_ml1_basketball |       |`bsub -n 4 -J "maml-tpro" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml1_basketball.py'` |
| maml_trpo_metaworld_ml10 | 35min      |`bsub -n 4 -J "maml-tpro" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml10.py'` |
| maml_trpo_metaworld_ml45 | 50min      |`bsub -n 15 -J "maml-tpro" -W 24:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml45.py'` |
| pearl_metaworld_ml1_basketball |            |`bsub -n 4 -J "pearl" -W 300:00 -R "rusage[mem=4096]" 'python src/pearl_metaworld_ml1_basketball.py'`|
| pearl_metaworld_ml10     |            |`bsub -n 4 -J "pearl" -W 24:00 -R "rusage[mem=4096]" 'python src/pearl_metaworld_ml10.py'`|
| pearl_metaworld_ml10 gpu |            |`bsub -n 10 -J "pearl" -W 24:00 -R "rusage[mem=2048, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX1080Ti]" 'python src/pearl_metaworld_ml10.py' --use_gpu True`|

#### hyperparameter MAML ML1 - Basketball-v2 (Andreas)
| experiment               |cmd                                                                                                 | state |
|--------------------------|----------------------------------------------------------------------------------------------------|-------|
| outer_lr=5e-3            | `bsub -n 4 -J "maml-ml1-5e-3" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml1_basketball.py --outer_lr 5e-3'` |run|
| outer_lr=1e-3 default    | `bsub -n 4 -J "maml-ml1-1e-3" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml1_basketball.py --outer_lr 1e-3'` |run|
| outer_lr=5e-4            | `bsub -n 4 -J "maml-ml1-5e-4" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml1_basketball.py --outer_lr 5e-4'` |run|
| inner_lr=0.1 default     | `bsub -n 4 -J "maml-ml1-in-0.1" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml1_basketball.py --inner_lr 1e-1'`  |run|
| inner_lr=0.05            | `bsub -n 4 -J "maml-ml1-in-0.05" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml1_basketball.py --inner_lr 5e-2'` |run|
| inner_lr=0.01            | `bsub -n 4 -J "maml-ml1-in-0.01" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml1_basketball.py --inner_lr 1e-2'` |run|

| experiment           |cmd                                                                                                 | state |
|----------------------|----------------------------------------------------------------------------------------------------|-------|
| episodes_per_task=10 (default) | `bsub -n 10 -J "maml-ml1-eps-10" -W 300:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml1_basketball.py --episodes_per_task 10'`| | 
| episodes_per_task=20 | `bsub -n 10 -J "maml-ml1-eps-20" -W 300:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml1_basketball.py --episodes_per_task 20'`| | 
| episodes_per_task=45 | `bsub -n 10 -J "maml-ml1-eps-45" -W 300:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml1_basketball.py --episodes_per_task 45'`| |
| episodes_per_task=60 | `bsub -n 10 -J "maml-ml1-eps-60" -W 300:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml1_basketball.py --episodes_per_task 60'`| |

#### hyperparameter MAML ML10 (Jona)

| experiment               |cmd                                                                                                 | state |
|--------------------------|----------------------------------------------------------------------------------------------------|-------|
| outer_lr=1e-2            | `bsub -n 10 -J "maml-ml10-1e-2" -W 120:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml10.py --outer_lr 1e-2'` |run |
| outer_lr=5e-3            | `bsub -n 10 -J "maml-ml10-5e-3" -W 120:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml10.py --outer_lr 5e-3'` |run |
| outer_lr=1e-3 default    | `bsub -n 10 -J "maml-ml10-1e-3" -W 120:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml10.py --outer_lr 1e-3'` |run |
| outer_lr=5e-4            | `bsub -n 10 -J "maml-ml10-5e-4" -W 120:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml10.py --outer_lr 5e-4'` |run |
| inner_lr=0.5             | `bsub -n 10 -J "maml-ml10-in-0.5" -W 120:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml10.py --inner_lr 0.5'`  | run
| inner_lr=0.1 default     | `bsub -n 10 -J "maml-ml10-in-0.1" -W 120:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml10.py --inner_lr 0.1'`  | run
| inner_lr=0.05            | `bsub -n 10 -J "maml-ml10-in-0.05" -W 120:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml10.py --inner_lr 0.05'` | run
| inner_lr=0.01            | `bsub -n 10 -J "maml-ml10-in-0.01" -W 120:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml10.py --inner_lr 0.01'` | run

| experiment               |cmd                                                                                                 | state |
|--------------------------|----------------------------------------------------------------------------------------------------|-------|
| episodes_per_task=10 (default) | `bsub -n 10 -J "maml-ml10-eps-10" -W 300:00 -R "rusage[mem=2048]" 'python src/maml_trpo_metaworld_ml10.py --episodes_per_task 10'` | run |
| episodes_per_task=20 | `bsub -n 10 -J "maml-ml10-eps-20" -W 120:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml10.py --episodes_per_task 20'` | run |
| episodes_per_task=45 | `bsub -n 10 -J "maml-ml10-eps-45" -W 120:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml10.py --episodes_per_task 45'` | run |
| episodes_per_task=60 | `bsub -n 10 -J "maml-ml10-eps-60" -W 120:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml10.py --episodes_per_task 60'` | run |

#### hyperparameter MAML ML45 (Andreas)

| experiment               |cmd                                                                                                 | state |
|--------------------------|----------------------------------------------------------------------------------------------------|-------|
| outer_lr=5e-3            | `bsub -n 20 -J "maml45-5e-3" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml45.py --outer_lr 5e-3'` |run|
| outer_lr=1e-3 default    | `bsub -n 20 -J "maml45-1e-3" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml45.py --outer_lr 1e-3'` |run|
| outer_lr=5e-4            | `bsub -n 20 -J "maml45-5e-4" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml45.py --outer_lr 5e-4'` |run|
| inner_lr=0.1 default     | `bsub -n 20 -J "maml45-in-0.1" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml45.py --inner_lr 0.1'`  |run|
| inner_lr=0.05            | `bsub -n 20 -J "maml45-in-0.05" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml45.py --inner_lr 0.05'` |run|
| inner_lr=0.01            | `bsub -n 20 -J "maml45-in-0.01" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml45.py --inner_lr 0.01'` |run|

| experiment               |cmd                                                                                                 | state |
|--------------------------|----------------------------------------------------------------------------------------------------|-------|
| episodes_per_task=10     | `bsub -n 20 -J "maml-ml45-eps-10" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml45.py --episodes_per_task 10'` | |
| episodes_per_task=20     | `bsub -n 20 -J "maml-ml45-eps-20" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml45.py --episodes_per_task 20'` | |
| episodes_per_task=45 (default)    | `bsub -n 20 -J "maml-ml45-eps-45" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml45.py --episodes_per_task 45'` | |
| episodes_per_task=60     | `bsub -n 20 -J "maml-ml45-eps-60" -W 300:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml45.py --episodes_per_task 60'` | |

#### hyperparameter PEARL ML10 (Jona)

| experiment               |cmd                                                                                                         | state |
|--------------------------|------------------------------------------------------------------------------------------------------------|-------|
| lr=7e-4                  | `bsub -n 4 -J "pearl-ml10-7e-4" -W 300:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python src/pearl_metaworld_ml10.py --lr 7e-4 --use_gpu True'` | run |
| lr=3e-4 default          | `bsub -n 4 -J "pearl-ml10-3e-4" -W 300:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python src/pearl_metaworld_ml10.py --lr 3e-4 --use_gpu True'` | run |
| lr=1e-4                  | `bsub -n 4 -J "pearl-ml10-1e-4" -W 300:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python src/pearl_metaworld_ml10.py --lr 1e-4 --use_gpu True'` | run |

#### hyperparameter PEARL ML1 - Basketball-v2 (Jona)

| experiment               |cmd                                                                                                                   | state |
|--------------------------|----------------------------------------------------------------------------------------------------------------------|-------|
| lr=7e-4                  | `bsub -n 4 -J "pearl-ml1-7e-4" -W 300:00 -R "rusage[mem=4096,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python src/pearl_metaworld_ml1_basketball.py --lr 7e-4 --use_gpu True'` | run |
| lr=3e-4 default          | `bsub -n 4 -J "pearl-ml1-3e-4" -W 300:00 -R "rusage[mem=4096,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python src/pearl_metaworld_ml1_basketball.py --lr 3e-4 --use_gpu True'` | run |
| lr=1e-4                  | `bsub -n 4 -J "pearl-ml1-1e-4" -W 300:00 -R "rusage[mem=4096,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python src/pearl_metaworld_ml1_basketball.py --lr 1e-4 --use_gpu True'` | run |

#### hyperparameter PEARL ML45 (Andreas)

| experiment               |cmd                                                                                                         | state |
|--------------------------|------------------------------------------------------------------------------------------------------------|-------|
| lr=7e-4                  | `bsub -n 20 -J "pearl-ml45-7e-4" -W 300:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python src/pearl_metaworld_ml45.py --lr 7e-4 --use_gpu True'` | |
| lr=3e-4 default          | `bsub -n 20 -J "pearl-ml45-3e-4" -W 300:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python src/pearl_metaworld_ml45.py --lr 3e-4 --use_gpu True'` | queue |
| lr=1e-4                  | `bsub -n 20 -J "pearl-ml45-1e-4" -W 300:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python src/pearl_metaworld_ml45.py --lr 1e-4 --use_gpu True'` | |

### cpu smaller job (10*3Gb)
```
bsub -n 10 -J "maml-tpro" -W 4:00 -R "rusage[mem=3072]" 'python src/maml_trpo_metaworld_ml10.py'
```
### cpu larger job (20*4Gb)
```
bsub -n 20 -J "maml-tpro" -W 24:00 -R "rusage[mem=4096]" 'python src/maml_trpo_metaworld_ml10.py'
```

### gpu smaller job (10*3Gb & any gpu)
```
bsub -n 10 -J "maml-tpro" -W 4:00 -R "rusage[mem=3072, ngpus_excl_p=1]" 'python src/maml_trpo_metaworld_ml10.py'
```

### gpu larger job (20*4Gb & 2080Ti)
```
bsub -n 20 -J "maml-tpro" -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python src/maml_trpo_metaworld_ml10.py'
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
