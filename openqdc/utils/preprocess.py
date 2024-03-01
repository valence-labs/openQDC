import click
import numpy as np
from loguru import logger
from openqdc import AVAILABLE_DATASETS

options = list(AVAILABLE_DATASETS.values())
options_map = {d.__name__: d for d in options}


@click.command()
@click.option("--dataset", "-d", type=str, default="ani1", help="Dataset name or index.")
def preprocess(dataset):
    if dataset not in options_map:
        dataset_id = int(dataset)
        data_class = options[dataset_id]
    else:
        data_class = options_map[dataset]

    data_class.no_init().preprocess(overwrite=False)
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
