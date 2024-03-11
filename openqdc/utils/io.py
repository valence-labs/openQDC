"""IO utilities."""

import json
import os
import pickle as pkl

import fsspec
import h5py
from aiohttp import ClientTimeout
from ase.atoms import Atoms
from fsspec.callbacks import TqdmCallback
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from rdkit.Chem import MolFromXYZFile

gcp_filesys = fsspec.filesystem("gs")  # entry point for google bucket (need gsutil permission)
gcp_filesys_public = fsspec.filesystem("https")  # public API for download
local_filesys = LocalFileSystem()

gcp_filesys_public.client_kwargs = {"timeout": ClientTimeout(total=3600, connect=1000)}

_OPENQDC_CACHE_DIR = (
    "~/.cache/openqdc" if "OPENQDC_CACHE_DIR" not in os.environ else os.path.normpath(os.environ["OPENQDC_CACHE_DIR"])
)


def set_cache_dir(d):
    r"""
    Optionally set the _OPENQDC_CACHE_DIR directory.

    Args:
        d (str): path to a local folder.
    """
    if d is None:
        return
    global _OPENQDC_CACHE_DIR
    _OPENQDC_CACHE_DIR = os.path.normpath(os.path.expanduser(d))


def get_local_cache() -> str:
    """
    Returns the local cache directory. It creates it if it does not exist.

    Returns:
        str: path to the local cache directory
    """
    cache_dir = os.path.expanduser(os.path.expandvars(_OPENQDC_CACHE_DIR))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_remote_cache(write_access=False) -> str:
    """
    Returns the entry point based on the write access.
    """
    if write_access:
        remote_cache = "gs://qmdata-public/openqdc"
    else:
        remote_cache = "https://storage.googleapis.com/qmdata-public/openqdc"
    return remote_cache


def push_remote(local_path, overwrite=True):
    """
    Attempt to push file to remote gs path
    """
    remote_path = local_path.replace(get_local_cache(), get_remote_cache(write_access=overwrite))
    gcp_filesys.mkdirs(os.path.dirname(remote_path), exist_ok=False)
    print(f"Pushing {local_path} file to {remote_path}, ({gcp_filesys.exists(os.path.dirname(remote_path))})")
    if not gcp_filesys.exists(remote_path) or overwrite:
        gcp_filesys.put_file(
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
    return remote_path


def pull_locally(local_path, overwrite=False):
    """
    Retrieve file from remote gs path or local cache
    """
    remote_path = local_path.replace(get_local_cache(), get_remote_cache())
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path) or overwrite:
        gcp_filesys_public.get_file(
            remote_path,
            local_path,
            callback=TqdmCallback(
                tqdm_kwargs={
                    "ascii": " ▖▘▝▗▚▞-",
                    "desc": f"Downloading {os.path.basename(remote_path)}",
                    "unit": "B",
                },
            ),
        )
    return local_path


def copy_exists(local_path):
    remote_path = local_path.replace(get_local_cache(), get_remote_cache())
    return os.path.exists(local_path) or gcp_filesys_public.exists(remote_path)


def makedirs_gcs(path, exist_ok=True):
    """Creates directory"""
    fs: GCSFileSystem = fsspec.filesystem("gs")
    fs.mkdirs(path, exist_ok=exist_ok)


def makedirs(path, exist_ok=True):
    os.makedirs(path, exist_ok=exist_ok)


def check_file(path) -> bool:
    """Checks if file present on local"""
    return os.path.exists(path)


def check_file_gcs(path) -> bool:
    """Checks if file present on GCS FileSystem"""
    # get file system
    fs: GCSFileSystem = fsspec.filesystem("gs")
    return fs.exists(path)


def save_pkl(file, path):
    """Saves pkl file"""
    print(f"Saving file at {path}")
    with fsspec.open(path, "wb") as fp:  # Pickling
        pkl.dump(file, fp)
    print("Done")


def load_pkl_gcs(path, check=True):
    """Load pkl file from GCS FileSystem"""
    if check:
        if not check_file_gcs(path):
            raise FileNotFoundError(f"File {path} does not exist on GCS and local.")

    # get file system
    fs: GCSFileSystem = fsspec.filesystem("gs")

    with fs.open(path, "rb") as fp:  # Unpickling
        return pkl.load(fp)


def load_pkl(path, check=True):
    """Load pkl file"""
    if check:
        if not check_file(path):
            raise FileNotFoundError(f"File {path} does not exist on GCS and local.")

    with open(path, "rb") as fp:  # Unpickling
        return pkl.load(fp)


def load_hdf5_file(hdf5_file_path: str):
    """Loads hdf5 file with fsspec"""
    if not check_file(hdf5_file_path):
        raise FileNotFoundError(f"File {hdf5_file_path} does not exist on GCS and local.")

    fp = fsspec.open(hdf5_file_path, "rb")
    if hasattr(fp, "open"):
        fp = fp.open()
    file = h5py.File(fp)

    # inorder to enable multiprocessing:
    # https://github.com/fsspec/gcsfs/issues/379#issuecomment-839929801
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None

    return file


def create_hdf5_file(hdf5_file_path: str):
    """Creates hdf5 file with fsspec"""
    fp = fsspec.open(hdf5_file_path, "wb")
    if hasattr(fp, "open"):
        fp = fp.open()
    return h5py.File(fp, "a")


def load_json(path):
    """Loads json file"""
    with fsspec.open(path, "r") as fp:  # Unpickling
        return json.load(fp)


def load_xyz(path):
    """
    Load XYZ file using RDKit
    """
    return MolFromXYZFile(path)


def dict_to_atoms(d: dict, ext: bool = False) -> Atoms:
    """
    Converts dictionary to ase atoms object

    Args:
        d (dict): dictionary containing keys: positions, atomic_numbers, charges
        ext (bool, optional): Whether to include all the rest of the dictionary in the atoms object info field.
        Defaults to False.
    """
    pos, atomic_numbers, charges = d.pop("positions"), d.pop("atomic_numbers"), d.pop("charges")
    at = Atoms(positions=pos, numbers=atomic_numbers, charges=charges)
    if ext:
        at.info = d
    return at


def print_h5_tree(val, pre=""):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + "└── " + key)
                print_h5_tree(val, pre + "    ")
            else:
                print(pre + "└── " + key + " (%d)" % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + "├── " + key)
                print_h5_tree(val, pre + "│   ")
            else:
                # pass
                print(pre + "├── " + key + " (%d)" % len(val))
