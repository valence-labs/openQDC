from typing import List, Optional

import typer
from loguru import logger
from prettytable import PrettyTable
from rich import print
from typing_extensions import Annotated

from openqdc.datasets import COMMON_MAP_POTENTIALS  # noqa
from openqdc.datasets import (
    AVAILABLE_DATASETS,
    AVAILABLE_INTERACTION_DATASETS,
    AVAILABLE_POTENTIAL_DATASETS,
)

app = typer.Typer(help="OpenQDC CLI")


def sanitize(dictionary):
    return {k.lower().replace("_", "").replace("-", ""): v for k, v in dictionary.items()}


SANITIZED_AVAILABLE_DATASETS = sanitize(AVAILABLE_DATASETS)


def exist_dataset(dataset):
    if dataset not in sanitize(AVAILABLE_DATASETS):
        logger.error(f"{dataset} is not available. Please open an issue on Github for the team to look into it.")
        return False
    return True


def format_entry(empty_dataset):
    energy_methods = [str(x) for x in empty_dataset.__energy_methods__]
    max_num_to_display = 6
    if len(energy_methods) > 6:
        entry = ",".join(energy_methods[:max_num_to_display]) + "..."
    else:
        entry = ",".join(energy_methods[:max_num_to_display])
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
            if SANITIZED_AVAILABLE_DATASETS[dataset].no_init().is_cached() and not overwrite:
                logger.info(f"{dataset} is already cached. Skipping download")
            else:
                SANITIZED_AVAILABLE_DATASETS[dataset](overwrite_local_cache=True, cache_dir=cache_dir)


@app.command()
def datasets():
    """
    Print a table of the available openQDC datasets and some informations.
    """
    table = PrettyTable(["Name", "Type of Energy", "Forces", "Level of theory"])
    for dataset in AVAILABLE_DATASETS:
        empty_dataset = AVAILABLE_DATASETS[dataset].no_init()
        has_forces = False if not any(empty_dataset.force_mask) else True
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
def fetch(
    datasets: List[str],
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite or force the re-download of the files.",
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
    Download the raw datasets files from the main openQDC hub.
    overwrite: bool = False,
        If True, the files will be re-downloaded and overwritten.
    cache_dir: Optional[str] = None,
        Path to the cache. If not provided, the default cache directory will be used.
    Special case: if the dataset is "all", "potential", "interaction".
        all: all available datasets will be downloaded.
        potential: all the potential datasets will be downloaded
        interaction: all the interaction datasets will be downloaded
    Example:
        openqdc fetch Spice
    """
    if datasets[0].lower() == "all":
        dataset_names = list(sanitize(AVAILABLE_DATASETS).keys())
    elif datasets[0].lower() == "potential":
        dataset_names = list(sanitize(AVAILABLE_POTENTIAL_DATASETS).keys())
    elif datasets[0].lower() == "interaction":
        dataset_names = list(sanitize(AVAILABLE_INTERACTION_DATASETS).keys())
    else:
        dataset_names = datasets
    for dataset in list(map(lambda x: x.lower().replace("_", ""), dataset_names)):
        if exist_dataset(dataset):
            try:
                SANITIZED_AVAILABLE_DATASETS[dataset].fetch(cache_dir, overwrite)
            except Exception as e:
                logger.error(f"Something unexpected happended while fetching {dataset}: {repr(e)}")


@app.command()
def preprocess(
    datasets: List[str],
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite or force the re-download of the datasets.",
        ),
    ] = True,
    upload: Annotated[
        bool,
        typer.Option(
            help="Whether to try the upload to the remote storage.",
        ),
    ] = False,
):
    """
    Preprocess a raw dataset (previously fetched) into a openqdc dataset and optionally push it to remote.
    """
    for dataset in list(map(lambda x: x.lower().replace("_", ""), datasets)):
        if exist_dataset(dataset):
            logger.info(f"Preprocessing {SANITIZED_AVAILABLE_DATASETS[dataset].__name__}")
            try:
                SANITIZED_AVAILABLE_DATASETS[dataset].no_init().preprocess(upload=upload, overwrite=overwrite)
            except Exception as e:
                logger.error(f"Error while preprocessing {dataset}. {e}. Did you fetch the dataset first?")
                raise e


@app.command()
def upload(
    datasets: List[str],
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite or force the re-download of the datasets.",
        ),
    ] = True,
):
    """
    Upload a preprocessed dataset to the remote storage.
    """
    for dataset in list(map(lambda x: x.lower().replace("_", ""), datasets)):
        if exist_dataset(dataset):
            logger.info(f"Uploading {SANITIZED_AVAILABLE_DATASETS[dataset].__name__}")
            try:
                SANITIZED_AVAILABLE_DATASETS[dataset]().upload(overwrite=overwrite)
            except Exception as e:
                logger.error(f"Error while uploading {dataset}. {e}. Did you preprocess the dataset first?")
                raise e


if __name__ == "__main__":
    app()
