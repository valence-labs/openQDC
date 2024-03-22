from typing import List, Optional

import typer
from loguru import logger
from prettytable import PrettyTable
from typing_extensions import Annotated

from openqdc import AVAILABLE_DATASETS, AVAILABLE_POTENTIAL_DATASETS
from openqdc.raws.config_factory import DataConfigFactory, DataDownloader

app = typer.Typer(help="OpenQDC CLI")


def exist_dataset(dataset):
    if dataset not in AVAILABLE_DATASETS:
        logger.error(f"{dataset} is not available. Please open an issue on Github for the team to look into it.")
        return False
    return True


def format_entry(empty_dataset):
    if len(empty_dataset.__energy_methods__) > 10:
        entry = ",".join(empty_dataset.__energy_methods__[:10]) + "..."
    else:
        entry = ",".join(empty_dataset.__energy_methods__[:10])
    return entry


@app.command()
def download(
    datasets: List[str],
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite or force the re-download of the datasets.",
        ),
    ] = False,
    cache_dir: Annotated[
        Optional[str],
        typer.Option(
            help="Path to the cache. If not provided, the default cache directory (.cache/openqdc/) will be used.",
        ),
    ] = None,
):
    """
    Download preprocessed ml-ready datasets from the main openQDC hub.

    Example:
        openqdc download Spice QMugs
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
    Print a table of the available openQDC datasets and some informations.
    """
    table = PrettyTable(["Name", "Type of Energy", "Forces", "Level of theory"])
    for dataset in AVAILABLE_DATASETS:
        empty_dataset = AVAILABLE_DATASETS[dataset].no_init()
        has_forces = False if not empty_dataset.__force_methods__ else True
        en_type = "Potential" if dataset in AVAILABLE_POTENTIAL_DATASETS else "Interaction"
        table.add_row(
            [
                dataset,
                en_type,
                has_forces,
                format_entry(empty_dataset),
            ]
        )
    table.align = "l"
    print(table)


@app.command()
def fetch(datasets: List[str]):
    """
    Download the raw datasets files from the main openQDC hub.
    Special case: if the dataset is "all", all available datasets will be downloaded.

    Example:
        openqdc fetch Spice
    """
    if datasets[0] == "all":
        dataset_names = DataConfigFactory.available_datasets
    else:
        dataset_names = datasets

    for dataset_name in dataset_names:
        dd = DataDownloader()
        dd.from_name(dataset_name)


if __name__ == "__main__":
    app()
