import click
from garage import Trainer, wrap_experiment
from garage.experiment.deterministic import set_seed

from helpers import out_dir_config
from helpers import environmentvariables

# Init env. variables
environmentvariables.initialize()


@click.command()
@click.option('--seed', default=1)
@click.option('--resume_from_dir', default='')  # e.g. "./data/local/experiment/maml_trpo_metaworld_ml10"
@wrap_experiment(snapshot_mode='all', log_dir=out_dir_config.get_out_dir(__file__))
def continue_training(ctxt, seed=1, resume_from_dir=""):
    set_seed(seed)
    trainer = Trainer(ctxt)
    trainer.restore(resume_from_dir)
    trainer.resume()


continue_training()
