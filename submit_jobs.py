"""
Main class to speed ub job creation on the cluster.
"""
import itertools
import os

import click

algos_file = {
    # algo name - file name component
    "maml": 'maml_trpo',
    "pearl": 'pearl'
}

envs_file = {
    # env name - file name component
    "ml1": 'metaworld_ml1_basketball.py',
    "ml10": 'metaworld_ml10.py',
    "ml45": 'metaworld_ml45.py'
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
