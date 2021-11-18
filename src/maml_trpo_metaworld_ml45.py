#!/usr/bin/env python3
"""This is an example to train MAML-TRPO on ML45 environment."""
# pylint: disable=no-value-for-parameter
# yapf: disable
import click
import metaworld
import torch
from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment import (MetaWorldTaskSampler, SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from helpers import environmentvariables
from helpers import out_dir_config
from experiment.custom_meta_evaluator import CustomMetaEvaluator

# yapf: enable

# Init env. variables
environmentvariables.initialize()


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=300)
@click.option('--episodes_per_task', default=45)
@click.option('--meta_batch_size', default=45)
@click.option('--inner_lr', default=0.1)
@click.option('--outer_lr', default=1e-3)
@wrap_experiment(snapshot_mode='all', log_dir=out_dir_config.get_out_dir(__file__))
def maml_trpo_metaworld_ml45(ctxt, seed, epochs, episodes_per_task,
                             meta_batch_size,
                             inner_lr,
                             outer_lr):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        episodes_per_task (int): Number of episodes per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.
        inner_lr (float): Adaptation learning rate.
        outer_lr (float): Meta policy learning rate.
    """
    set_seed(seed)
    ml45 = metaworld.ML45()

    # pylint: disable=missing-return-doc,missing-return-type-doc
    def wrap(env, _):
        return normalize(env, expected_action_scale=10.0)

    train_task_sampler = MetaWorldTaskSampler(ml45, 'train', wrap)
    test_env = wrap(MetaWorldSetTaskEnv(ml45, 'test'), None)
    test_task_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                       env=test_env,
                                       wrapper=wrap)
    env = train_task_sampler.sample(45)[0]()

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 100),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    meta_evaluator = CustomMetaEvaluator(test_task_sampler=test_task_sampler)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length,
                         n_workers=meta_batch_size)

    trainer = Trainer(ctxt)
    algo = MAMLTRPO(env=env,
                    task_sampler=train_task_sampler,
                    policy=policy,
                    sampler=sampler,
                    value_function=value_function,
                    meta_batch_size=meta_batch_size,
                    discount=0.99,
                    gae_lambda=1.,
                    inner_lr=inner_lr,
                    outer_lr=outer_lr,
                    num_grad_updates=1,
                    meta_evaluator=meta_evaluator)

    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=episodes_per_task * env.spec.max_episode_length)


maml_trpo_metaworld_ml45()
