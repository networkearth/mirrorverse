import json

import click 

from mirrorverse import hypercube as hypercube_module

@click.group()
def cli():
    pass

@cli.command()
@click.option('--config', required=True, type=str)
def hypercube(config):
    with open(config, 'r') as fh:
        config = json.load(fh)
    hypercube_module.main(**config)
