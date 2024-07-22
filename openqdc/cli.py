import os
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
from openqdc.utils.io import get_local_cache

app = typer.Typer(help="OpenQDC CLI")


def sanitize(dictionary):
    """
    Sanitize dataset names to be used in the CLI.
    """
    return {k.lower().replace("_", "").replace("-", ""): v for k, v in dictionary.items()}


SANITIZED_AVAILABLE_DATASETS = sanitize(AVAILABLE_DATASETS)


def exist_dataset(dataset) -> bool:
    """
    Check if dataset is available in the openQDC datasets.
    """
    if dataset not in sanitize(AVAILABLE_DATASETS):
        logger.error(f"{dataset} is not available. Please open an issue on Github for the team to look into it.")
        return False
    return True


def format_entry(empty_dataset, max_num_to_display: int = 6):
    """
    Format the entry for the table.
    max_num_to_display: int = 6,
        Maximum number of energy methods to display. Used to keep the table format
        readable in case of datasets with many energy methods. [ex. MultiXQM9]
    """
    energy_methods = [str(x) for x in empty_dataset.__energy_methods__]

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
            help="Whether to force the re-download of the datasets and overwrite the current cached dataset.",
        ),
    ] = False,
    cache_dir: Annotated[
        Optional[str],
        typer.Option(
            help="Path to the cache. If not provided, the default cache directory (.cache/openqdc/) will be used.",
        ),
    ] = None,
    as_zarr: Annotated[
        bool,
        typer.Option(
            help="Whether to use a zarr format for the datasets instead of memmap.",
        ),
    ] = False,
    gs: Annotated[
        bool,
        typer.Option(
            help="Whether source to use for downloading. If True, Google Storage will be used."
            + "Otherwise, AWS S3 will be used",
        ),
    ] = False,
):
    """
    Download preprocessed ml-ready datasets from the main openQDC hub.

    Example:
        openqdc download Spice QMugs
    """
    if gs:
        os.environ["OPENQDC_DOWNLOAD_API"] = "gs"

    for dataset in list(map(lambda x: x.lower().replace("_", ""), datasets)):
        if exist_dataset(dataset):
            ds = SANITIZED_AVAILABLE_DATASETS[dataset].no_init()
            ds.read_as_zarr = as_zarr
            if ds.is_cached() and not overwrite:
                logger.info(f"{dataset} is already cached. Skipping download")
            else:
                SANITIZED_AVAILABLE_DATASETS[dataset](
                    overwrite_local_cache=True, cache_dir=cache_dir, read_as_zarr=as_zarr, skip_statistics=True
                )


@app.command()
def datasets():
    """
    Print a formatted table of the available openQDC datasets and some informations.
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
            help="Whether to overwrite or force the re-download of the raw files.",
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
    Download the raw datasets files from the main openQDC hub.\n
    Special case: if the dataset is "all", "potential", "interaction".\n
    all: all available datasets will be downloaded.\n
    potential: all the potential datasets will be downloaded\n
    interaction: all the interaction datasets will be downloaded\n\n

    Example:\n
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
            help="Whether to overwrite the current cached datasets.",
        ),
    ] = True,
    upload: Annotated[
        bool,
        typer.Option(
            help="Whether to attempt the upload to the remote storage. Must have write permissions.",
        ),
    ] = False,
    as_zarr: Annotated[
        bool,
        typer.Option(
            help="Whether to preprocess as a zarr format or a memmap format.",
        ),
    ] = False,
):
    """
    Preprocess a raw dataset (previously fetched) into a openqdc dataset and optionally push it to remote.

    Example:
        openqdc preprocess Spice QMugs
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
            help="Whether to overwrite the remote files if they are present.",
        ),
    ] = True,
    as_zarr: Annotated[
        bool,
        typer.Option(
            help="Whether to upload the zarr files if available.",
        ),
    ] = False,
):
    """
    Upload a preprocessed dataset to the remote storage.

    Example:
        openqdc upload Spice --overwrite
    """
    for dataset in list(map(lambda x: x.lower().replace("_", ""), datasets)):
        if exist_dataset(dataset):
            logger.info(f"Uploading {SANITIZED_AVAILABLE_DATASETS[dataset].__name__}")
            try:
                SANITIZED_AVAILABLE_DATASETS[dataset](skip_statistics=True).upload(overwrite=overwrite, as_zarr=as_zarr)
            except Exception as e:
                logger.error(f"Error while uploading {dataset}. {e}. Did you preprocess the dataset first?")
                raise e


@app.command()
def convert(
    datasets: List[str],
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite the current zarr cached datasets.",
        ),
    ] = False,
    download: Annotated[
        bool,
        typer.Option(
            help="Whether to force the re-download of the memmap datasets.",
        ),
    ] = False,
):
    """
    Convert a preprocessed dataset from a memmap dataset to a zarr dataset.
    """
    import os
    from os.path import join as p_join

    import numpy as np
    import zarr

    from openqdc.utils.io import load_pkl

    def silent_remove(filename):
        """
        Zarr zip files are currently not overwritable. This function is used to remove the file if it exists.
        """
        try:
            os.remove(filename)
        except OSError:
            pass

    for dataset in list(map(lambda x: x.lower().replace("_", ""), datasets)):
        if exist_dataset(dataset):
            logger.info(f"Converting {SANITIZED_AVAILABLE_DATASETS[dataset].__name__}")
            try:
                ds = SANITIZED_AVAILABLE_DATASETS[dataset](overwrite_local_cache=download, skip_statistics=True)
                # os.makedirs(p_join(ds.root, "zips", ds.__name__), exist_ok=True)

                pkl = load_pkl(p_join(ds.preprocess_path, "props.pkl"))
                metadata = p_join(ds.preprocess_path, "metadata.zip")
                if overwrite:
                    silent_remove(metadata)
                group = zarr.group(zarr.storage.ZipStore(metadata))
                for key, value in pkl.items():
                    # sub=group.create_group(key)
                    if key in ["name", "subset"]:
                        data = group.create_dataset(key, shape=value[0].shape, dtype=value[0].dtype)
                        data[:] = value[0][:]
                        data2 = group.create_dataset(key + "_ptr", shape=value[1].shape, dtype=np.int32)
                        data2[:] = value[1][:]
                    else:
                        data = group.create_dataset(key, shape=value.shape, dtype=value.dtype)
                        data[:] = value[:]

                force_attrs = {
                    "unit": str(ds.force_unit),
                    "level_of_theory": ds.force_methods,
                }

                energy_attrs = {"unit": str(ds.energy_unit), "level_of_theory": ds.energy_methods}

                atomic_inputs_attrs = {
                    "unit": str(ds.distance_unit),
                }
                attrs = {"forces": force_attrs, "energies": energy_attrs, "atomic_inputs": atomic_inputs_attrs}

                # os.makedirs(p_join(ds.root, "zips",  ds.__name__), exist_ok=True)
                for key, value in ds.data.items():
                    if key not in ds.data_keys:
                        continue
                    print(key, value.shape)

                    zarr_path = p_join(ds.preprocess_path, key + ".zip")  # ds.__name__,
                    if overwrite:
                        silent_remove(zarr_path)
                    z = zarr.open(
                        zarr.storage.ZipStore(zarr_path), "w", zarr_version=2, shape=value.shape, dtype=value.dtype
                    )
                    z[:] = value[:]
                    if key in attrs:
                        z.attrs.update(attrs[key])

            except Exception as e:
                logger.error(f"Error while converting {dataset}. {e}. Did you preprocess the dataset first?")
                raise e


@app.command()
def cache():
    """
    Get the current local cache path of openQDC
    """
    print(f"openQDC local cache:\n {get_local_cache()}")


if __name__ == "__main__":
    app()
