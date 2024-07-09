import gzip
import os
import shutil
import socket
import tarfile
import urllib.error
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from typing import Optional

import fsspec
import gdown
import requests
import tqdm

# from aiohttp import ClientTimeout
from dotenv import load_dotenv
from fsspec import AbstractFileSystem
from fsspec.callbacks import TqdmCallback
from fsspec.implementations.local import LocalFileSystem
from loguru import logger
from sklearn.utils import Bunch

import openqdc.utils.io as ioqdc
import os 

@dataclass
class FileSystem:
    """
    A class to handle file system operations
    """

    public_endpoint: Optional[AbstractFileSystem] = None
    private_endpoint: Optional[AbstractFileSystem] = None
    local_endpoint: AbstractFileSystem = LocalFileSystem()
    endpoint_url = "https://874f02b9d981bd6c279e979c0d91c4b4.r2.cloudflarestorage.com"

    def __init__(self):
        load_dotenv()
        self.KEY = os.getenv("CLOUDFARE_KEY", None)
        self.SECRET = os.getenv("CLOUDFARE_SECRET", None)

    @property
    def public(self):
        self.connect()
        return self.public_endpoint

    @property
    def private(self):
        self.connect()
        return self.private_endpoint

    @property
    def local(self):
        return self.local_endpoint

    @property
    def is_connected(self):
        """
        Check if it is connected to the public or the private endpoints
        """
        return self.public_endpoint is not None or self.private_endpoint is not None

    def connect(self):
        """
        Attempt connection to the public and private endpoints
        """
        if not self.is_connected:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # No quota warning
                self.public_endpoint = self.get_default_endpoint("public")
                self.private_endpoint = self.get_default_endpoint("private")
                # self.public_endpoint.client_kwargs = {"timeout": ClientTimeout(total=3600, connect=1000)}

    def get_default_endpoint(self, endpoint: str) -> AbstractFileSystem:
        """
        Return a default endpoint for the given str [public, private]
        """
        if endpoint == "private":
            #return fsspec.filesystem("gs")
            return fsspec.filesystem("s3", key=self.KEY, secret=self.SECRET, endpoint_url=self.endpoint_url)
        elif endpoint == "public":
            return fsspec.filesystem(
                "s3",
                key="a046308c078b0134c9e261aa91f63ab2",
                secret="d5b32f241ad8ee8d0a3173cd51b4f36d6869f168b21acef75f244a81dc10e1fb",
                endpoint_url=self.endpoint_url,
            ) if not os.environ["OPENQDC_DOWNLOAD_API"] == "gs" else fsspec.filesystem("https")
            #return fsspec.filesystem("https")
        else:
            return self.local_endpoint

    def get_file(self, remote_path: str, local_path: str):
        """
        Retrieve file from remote gs path or local cache
        """
        self.public.get_file(
            remote_path,
            local_path,
            callback=TqdmCallback(
                tqdm_kwargs={
                    "ascii": " ▖▘▝▗▚▞-",
                    "desc": f"Downloading {os.path.basename(remote_path)}",
                    "unit": "B",
                }
            ),
        )

    def put_file(self, local_path: str, remote_path: str):
        """
        Attempt to push file to remote gs path
        """
        self.private.put_file(
            local_path,
            remote_path,
            callback=TqdmCallback(
                tqdm_kwargs={
                    "ascii": " ▖▘▝▗▚▞-",
                    "desc": f"Uploading {os.path.basename(remote_path)}",
                    "unit": "B",
                },
            ),
        )

    def exists(self, path):
        """
        Check if file exists
        """
        return self.public.exists(path)

    def mkdirs(self, path, exist_ok=True):
        """Creates directory"""
        self.private.mkdirs(path, exist_ok=exist_ok)


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
            cache_path = ioqdc.get_local_cache()

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


API = FileSystem()