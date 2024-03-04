from typing import List, Optional

import typer
from loguru import logger
from openqdc.raws.config_factory import DataConfigFactory
from openqdc.raws.fetch import DataDownloader
from prettytable import PrettyTable
from typing_extensions import Annotated

from openqdc import AVAILABLE_DATASETS

app = typer.Typer(help="OpenQDC CLI")


def exist_dataset(dataset):
    if dataset not in AVAILABLE_DATASETS:
        logger.error(f"{dataset} is not available. Please open an issue on Github for the team to look into it.")
        return False
    return True


@app.command()
def download(
    datasets: List[str],
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite the datasets",
        ),
    ] = False,
    cache_dir: Annotated[
        Optional[str],
        typer.Option(
            help="Path to cache directory",
        ),
    ] = None,
):
    """
    Download preprocessed datasets from openQDC.
    """
    for dataset in list(map(lambda x: x.lower().replace("_", ""), datasets)):
        if exist_dataset(dataset):
            if AVAILABLE_DATASETS[dataset].no_init().is_cached() and not overwrite:
                logger.info(f"{dataset} is already cached. Skipping download")
            else:
                AVAILABLE_DATASETS[dataset](overwrite_local_cache=True, cache_dir=cache_dir)


@app.command()
def datasets():
    """
    Print the available datasets.
    """
    table = PrettyTable(["Name", "Forces", "Level of theory"])
    for dataset in AVAILABLE_DATASETS:
        empty_dataset = AVAILABLE_DATASETS[dataset].no_init()
        has_forces = False if not empty_dataset.__force_methods__ else True
        table.add_row([dataset, has_forces, ",".join(empty_dataset.__energy_methods__)])
    table.align = "l"
    print(table)


@app.command()
def fetch(datasets: List[str]):
    """
    Download the raw datasets files from openQDC.
    """
    if datasets[0] == "all":
        dataset_names = DataConfigFactory.available_datasets

    for dataset_name in dataset_names:
        dd = DataDownloader()
        dd.from_name(dataset_name)


if __name__ == "__main__":
    app()
