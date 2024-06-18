import gzip
import os
import shutil
import socket
import tarfile
import urllib.error
import urllib.request
import zipfile

import fsspec
import gdown
import requests
import tqdm
from loguru import logger
from sklearn.utils import Bunch

from openqdc.utils.io import get_local_cache


def download_url(url, local_filename):
    """
    Download a file from a url to a local file.
    Parameters
    ----------
        url : str
            URL to download from.
        local_filename : str
            Local path for destination.
    """
    logger.info(f"Url: {url}  File: {local_filename}")
    if "drive.google.com" in url:
        gdown.download(url, local_filename, quiet=False)
    elif "raw.github" in url:
        r = requests.get(url, allow_redirects=True)
        with open(local_filename, "wb") as f:
            f.write(r.content)
    else:
        r = requests.get(url, stream=True)
        with fsspec.open(local_filename, "wb") as f:
            for chunk in tqdm.tqdm(r.iter_content(chunk_size=16384)):
                if chunk:
                    f.write(chunk)


def decompress_tar_gz(local_filename):
    """
    Decompress a tar.gz file.
    Parameters
    ----------
        local_filename : str
            Path to local file to decompress.
    """
    parent = os.path.dirname(local_filename)
    with tarfile.open(local_filename) as tar:
        logger.info(f"Verifying archive extraction states: {local_filename}")
        all_names = tar.getnames()
        all_extracted = all([os.path.exists(os.path.join(parent, x)) for x in all_names])
        if not all_extracted:
            logger.info(f"Extracting archive: {local_filename}")
            tar.extractall(path=parent)
        else:
            logger.info(f"Archive already extracted: {local_filename}")


def decompress_zip(local_filename):
    """
    Decompress a zip file.
    Parameters
    ----------
        local_filename : str
            Path to local file to decompress.
    """
    parent = os.path.dirname(local_filename)

    logger.info(f"Verifying archive extraction states: {local_filename}")
    with zipfile.ZipFile(local_filename, "r") as zip_ref:
        all_names = zip_ref.namelist()
        all_extracted = all([os.path.exists(os.path.join(parent, x)) for x in all_names])
        if not all_extracted:
            logger.info(f"Extracting archive: {local_filename}")
            zip_ref.extractall(parent)
        else:
            logger.info(f"Archive already extracted: {local_filename}")


def decompress_gz(local_filename):
    """
    Decompress a gz file.
    Parameters
    ----------
        local_filename : str
            Path to local file to decompress.
    """
    logger.info(f"Verifying archive extraction states: {local_filename}")
    out_filename = local_filename.replace(".gz", "")
    if out_filename.endswith("hdf5"):
        out_filename = local_filename.replace("hdf5", "h5")

    all_extracted = os.path.exists(out_filename)
    if not all_extracted:
        logger.info(f"Extracting archive: {local_filename}")
        with gzip.open(local_filename, "rb") as f_in, open(out_filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        logger.info(f"Archive already extracted: {local_filename}")


def fetch_file(url, local_filename, overwrite=False):
    """
    Download a file from a url to a local file. Useful for big files.
    Parameters
    ----------
    url : str
        URL to download from.
    local_filename : str
        Local file to save to.
    overwrite : bool
        Whether to overwrite existing files.
    Returns
    -------
    local_filename : str
        Local file.
    """
    try:
        if os.path.exists(local_filename) and not overwrite:
            logger.info("File already exists, skipping download")
        else:
            download_url(url, local_filename)

        # decompress archive if necessary
        parent = os.path.dirname(local_filename)
        if local_filename.endswith("tar.gz"):
            decompress_tar_gz(local_filename)

        elif local_filename.endswith("zip"):
            decompress_zip(local_filename)

        elif local_filename.endswith(".gz"):
            decompress_gz(local_filename)

        elif local_filename.endswith("xz"):
            logger.info(f"Extracting archive: {local_filename}")
            os.system(f"cd {parent} && xz -d *.xz")

        else:
            pass

    except (socket.gaierror, urllib.error.URLError) as err:
        raise ConnectionError("Could not download {} due to {}".format(url, err))

    return local_filename


class DataDownloader:
    """Download data from a remote source.
    Parameters
    ----------
    cache_path : str
        Path to the cache directory.
    overwrite : bool
        Whether to overwrite existing files.
    """

    def __init__(self, cache_path=None, overwrite=False):
        if cache_path is None:
            cache_path = get_local_cache()

        self.cache_path = cache_path
        self.overwrite = overwrite

    def from_config(self, config: dict):
        b_config = Bunch(**config)
        data_path = os.path.join(self.cache_path, b_config.dataset_name)
        os.makedirs(data_path, exist_ok=True)

        logger.info(f"Downloading the {b_config.dataset_name} dataset")
        for local, link in b_config.links.items():
            outfile = os.path.join(data_path, local)
            fetch_file(link, outfile)
