# CLI for dataset downloading and uploading
You can quickly download, fetch, preprocess and upload openQDC datasets using the command line interface (CLI).

## Datasets
Print a formatted table of the available openQDC datasets and some informations.

Usage:

    openqdc datasets [OPTIONS]

Options:

    --help          Show this message and exit.

## Cache
Get the current local cache path of openQDC

Usage:

    openqdc cache [OPTIONS]

Options:

    --help          Show this message and exit.


## Download
Download preprocessed ml-ready datasets from the main openQDC hub.

Usage:

    openqdc download DATASETS... [OPTIONS]

Options:

    --help          Show this message and exit.
    --overwrite     Whether to force the re-download of the datasets and overwrite the current cached dataset. [default: no-overwrite]
    --cache-dir     Path to the cache. If not provided, the default cache directory (.cache/openqdc/) will be used. [default: None]
    --as-zarr       Whether to use a zarr format for the datasets instead of memmap. [default: no-as-zarr]
    --gs            Whether source to use for downloading. If True, Google Storage will be used.Otherwise, AWS S3 will be used [default: no-gs]

Example:

    openqdc download Spice

## Fetch
Download the raw datasets files from the main openQDC hub

Note:

    Special case: if the dataset is "all", "potential", "interaction".

Usage:

    openqdc fetch DATASETS... [OPTIONS]

Options:

    --help          Show this message and exit.
    --overwrite     Whether to overwrite or force the re-download of the raw files. [default: no-overwrite]
    --cache-dir     Path to the cache. If not provided, the default cache directory (.cache/openqdc/) will be used. [default: None]

Example:

    openqdc fetch Spice

## Preprocess
Preprocess a raw dataset (previously fetched) into a openqdc dataset and optionally push it to remote.

Usage:

    openqdc preprocess DATASETS... [OPTIONS]

Options:

    --help         Show this message and exit.
    --overwrite    Whether to overwrite the current cached datasets. [default: overwrite]
    --upload       Whether to attempt the upload to the remote storage. Must have write permissions. [default: no-upload]
    --as-zarr      Whether to preprocess as a zarr format or a memmap format. [default: no-as-zarr]

Example:

    openqdc preprocess Spice QMugs

## Upload
Upload a preprocessed dataset to the remote storage

Usage:

    openqdc upload DATASETS... [OPTIONS]

Options:

    --help          Show this message and exit.
    --overwrite     Whether to overwrite the remote files if they are present. [default: overwrite]
    --as-zarr       Whether to upload the zarr files if available. [default: no-as-zarr]

Example:

    openqdc upload Spice --overwrite

## Convert
Convert a preprocessed dataset from a memmap dataset to a zarr dataset.

Usage:

    openqdc convert DATASETS... [OPTIONS]

Options:

    --help          Show this message and exit.
    --overwrite     Whether to overwrite the current zarr cached datasets. [default: no-overwrite]
    --download      Whether to force the re-download of the memmap datasets. [default: no-download]
