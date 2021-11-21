# Commands to run experiments
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

