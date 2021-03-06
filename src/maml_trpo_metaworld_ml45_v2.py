#!/usr/bin/env python3
"""This is an example to train MAML-TRPO on ML45 environment."""
# pylint: disable=no-value-for-parameter
# yapf: disable
import sys

import click
import metaworld
import torch
from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import (MetaWorldTaskSampler,
                               SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.torch.policies import GaussianMLPPolicy
from garage.trainer import Trainer

from algos.maml_trpo_v2 import MAMLTRPO
from experiment.custom_meta_evaluator import CustomMetaEvaluator

# yapf: enable
from helpers import out_dir_config, environmentvariables

# Init env. variables
environmentvariables.initialize()


@click.command()
@click.option('--seed', type=int, default=1)
@click.option('--epochs', type=int, default=2000)
@click.option('--rollouts_per_task', type=int, default=20)
@click.option('--meta_batch_size', type=int, default=45)
@click.option('--inner_lr', default=1e-4, type=float)
@click.option('--outer_lr', default=1e-3, type=float)
@click.option('--discount', default=0.99, type=float)
@wrap_experiment(log_dir=out_dir_config.get_out_dir(__file__, ''.join(sys.argv[1:])), archive_launch_repo=False)
def maml_trpo_metaworld_ML45(ctxt, seed, epochs, rollouts_per_task,
                             meta_batch_size, inner_lr,
                             outer_lr,
                             discount
                             ):
    """Set up environment and algorithm and run the task.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        rollouts_per_task (int): Number of rollouts per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.
        inner_lr (float): learning rate to use for the inner TRPO agent.
            This hyperparameter is typically the one to tune when tuning
            your MAML.
        outer_lr (float): Meta policy learning rate.
        discount (float): Discount factor.
    """
    set_seed(seed)

    ML45 = metaworld.ML45()
    tasks = MetaWorldTaskSampler(ML45, 'train')
    env = tasks.sample(45)[0]()
    test_sampler = SetTaskSampler(
        MetaWorldSetTaskEnv,
        env=MetaWorldSetTaskEnv(ML45, 'test'),
    )

    policy = GaussianMLPPolicy(env_spec=env.spec,
                               hidden_sizes=(128, 128),
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=torch.tanh,
                               min_std=0.5,
                               max_std=1.5)

    value_function = LinearFeatureBaseline(env_spec=env.spec)

    meta_evaluator = CustomMetaEvaluator(test_task_sampler=test_sampler,
                                         n_exploration_eps=rollouts_per_task,
                                         n_test_tasks=None,
                                         n_test_episodes=10)
    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)

    trainer = Trainer(ctxt)
    algo = MAMLTRPO(
        env=env,
        policy=policy,
        sampler=sampler,
        task_sampler=tasks,
        value_function=value_function,
        meta_batch_size=meta_batch_size,
        discount=discount,
        gae_lambda=1.,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        num_grad_updates=1,
        meta_evaluator=meta_evaluator,
        entropy_method='max',
        policy_ent_coeff=5e-5,
        stop_entropy_gradient=True,
        center_adv=False,
        evaluate_every_n_epochs=10,
    )

    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=rollouts_per_task * env.spec.max_episode_length)


maml_trpo_metaworld_ML45()
