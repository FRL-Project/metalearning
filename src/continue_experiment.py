import click
from garage import Trainer, wrap_experiment
from garage.experiment.deterministic import set_seed


@click.command()
@click.option('--seed', default=1)
@click.option('--resume_from_dir', default='')  # e.g. "./data/local/experiment/maml_trpo_metaworld_ml10"
@wrap_experiment
def continue_training(ctxt, seed=1, resume_from_dir=""):
    set_seed(seed)
    trainer = Trainer(ctxt)
    trainer.restore(resume_from_dir)
    trainer.resume()


continue_training()
