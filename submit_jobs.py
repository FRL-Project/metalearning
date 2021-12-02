"""
Main class to speed ub job creation on the cluster.
"""
import itertools
import os

import click

algos_file = {
    # algo name - file name component
    "maml": 'maml_trpo',
    "pearl": 'pearl',
    "sac": 'sac'
}

envs_file = {
    # env name - file name component
    "ml1": 'metaworld_ml1_basketball.py',
    "ml10": 'metaworld_ml10.py',
    "ml45": 'metaworld_ml45.py',
    "-": 'metaworld.py'
}

experiments_params = {
    "maml": {
        "outer_lr": {
            "outer_lr": [1e-2, 1e-3, 5e-3, 5e-4],
        },
        "inner_lr": {
            "inner_lr": [0.5, 1e-1, 5e-2, 1e-2],
        },
        "dis": {
            "discount": [0.95, 0.99, 0.999],
        },
        "eps": {
            "episodes_per_task": [10, 20, 45, 60]
        }
    },
    "pearl": {
        "lr": {
            "lr": ['7e-4', '3e-4', '1e-4'],
            "use_gpu": [True]
        }
    },
    "sac": {
        "env_name": {
            "env_name":
                [
                    'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2',
                    'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2',
                    'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2',
                    'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2',
                    'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2',
                    'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2',
                    'pick-place-wall-v2',
                    'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2',
                    'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2',
                    'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2',
                    'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2'
                ]
        }
    }
}


def product_dict(**kwargs):
    # https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


@click.command()
@click.option('--algo', default="maml")
@click.option('--env', default="ml10")
@click.option('--exp', default="dis")
@click.option('--time', default="120:00")
@click.option('--n_cpus', default="10")
@click.option('--mem', default="2048")
@click.option('--gpu', default=None)
def submit_job(algo, env, exp, time, n_cpus, mem, gpu=None, path="./src"):
    python_file_name = algos_file[algo] + "_" + envs_file[env]
    python_file_path = os.path.join(path, python_file_name)

    # if python_file_name == 'sac_metaworld':
    #     tasks_list = open("./cfg_files/tasks", "r")
    #     experiment_parameter = tasks_list.read().split("\n")
    # else:
    experiment_parameter = experiments_params[algo][exp]

    for params in product_dict(**experiment_parameter):

        command = ''

        # use 4 cpus
        command += 'bsub -n ' + n_cpus
        command += ' -J "' + algo + '-' + env + '-' + exp + ':' + str(list(params.values())[0]) + '"'
        # job time
        command += ' -W ' + time
        # memory per cpu
        command += ' -R "rusage[mem=' + mem
        if gpu is None:
            command += ']"'
        elif gpu is not None:
            command += ', ngpus_excl_p=1]"'
            if gpu == 1:
                command += ' -R "select[gpu_model0==GeForceGTX1080Ti]"'
            elif gpu == 2:
                command += ' -R "select[gpu_model0==GeForceRTX2080Ti]"'
            else:
                command += ' -R "select[gpu_mtotal0>=10240]"'  # GPU memory more then 10GB

        command += ' "python ' + str(python_file_path)
        for par in params:
            command += ' ' + '--' + par + ' ' + str(params[par])
        command += '"'

        print(command)
        os.system(command)


submit_job()
