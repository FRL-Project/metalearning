# based on https://github.com/rlworkgroup/garage/pull/2287/files#diff-bb2d80290ec525d9afbe7851a1a04123b6e3a1d29b641a95292286fa5b7581bb
# !/usr/bin/env python3
"""This is an example to train a Meta-World Environment with SAC algorithm."""
import click
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import numpy as np
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import deterministic
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

from pretrain.buffer import ReplayBuffer

import sys
sys.path.append("..")
from helpers import environmentvariables
from helpers import out_dir_config


# Init env. variables
environmentvariables.initialize()

@click.command()
@click.option('--env_name', type=str, default='basketball-v2')
@click.option('--seed', type=int, default=1)
@click.option('--gpu', type=int, default=None)
@wrap_experiment(snapshot_mode='gap', snapshot_gap=50, name_parameters='all', log_dir=out_dir_config.get_out_dir(__file__))
def sac_metaworld(ctxt=None, env_name=None, gpu=None, seed=1):
    """Set up environment and algorithm and run the task.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        env_name (str): Name of Meta-World environment to initialize
            this experiment with.
        gpu (int): GPU id to use for training.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    not_in_mw = 'the env_name specified is not a metaworld environment'
    env_name = env_name + '-goal-observable'
    assert env_name in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, not_in_mw

    deterministic.set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    env = env_cls(seed=seed)
    env._freeze_rand_vec = False  # leads to a parametric variation on every reset
    max_path_length = env.max_path_length

    env = GymEnv(env, max_episode_length=max_path_length)
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = ReplayBuffer(size=int(5e6),
                                 obs_dim=env.spec.observation_space.shape[0],
                                 action_dim=env.spec.action_space.shape[0],
                                 log_dir=ctxt.snapshot_dir
                                 )
    batch_size = max_path_length
    num_evaluation_points = 500
    timesteps = int(5e6)
    epochs = timesteps // batch_size
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           n_workers=1,
                           worker_class=FragmentWorker,
                           worker_args=dict(n_envs=2))

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              gradient_steps_per_itr=batch_size,
              max_episode_length_eval=max_path_length,
              replay_buffer=replay_buffer,
              min_buffer_size=1e4,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=256,
              steps_per_epoch=epoch_cycles,
              num_evaluation_episodes=10,
              sampler=sampler)

    if gpu is not None:
        set_gpu_mode(True, gpu)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=epochs, batch_size=batch_size)
    replay_buffer.save(env_name + "buffer.h5")


sac_metaworld()
