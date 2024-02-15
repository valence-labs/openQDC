import click
import numpy as np
from loguru import logger

from openqdc import datasets

options = [
    datasets.ANI1,
    datasets.ANI1CCX,
    datasets.ANI1X,
    datasets.COMP6,
    datasets.DESS,
    datasets.GDML,
    datasets.GEOM,
    datasets.ISO17,
    datasets.Molecule3D,
    datasets.NablaDFT,
    datasets.OrbnetDenali,
    datasets.PCQM_B3LYP,
    datasets.PCQM_PM6,
    datasets.QM7X,
    datasets.QMugs,
    datasets.SN2RXN,
    datasets.SolvatedPeptides,
    datasets.Spice,
    datasets.TMQM,
    datasets.Transition1X,
    datasets.WaterClusters,
    datasets.MultixcQM9,
]

options_map = {d.__name__: d for d in options}


@click.command()
@click.option("--dataset", "-d", type=str, default="ani1", help="Dataset name or index.")
def preprocess(dataset):
    if dataset not in options_map:
        dataset_id = int(dataset)
        data_class = options[dataset_id]
    else:
        data_class = options_map[dataset]

    data_class().preprocess(overwrite=True)
    data = data_class()
    logger.info(f"Preprocessing {data.__name__}")

    n = len(data)
    for i in np.random.choice(n, 3, replace=False):
        x = data[i]
        print(x.name, x.subset, end=" ")
        for k in x:
            if isinstance(x[k], np.ndarray):
                print(k, x[k].shape, end=" ")
        print()


if __name__ == "__main__":
    preprocess()
