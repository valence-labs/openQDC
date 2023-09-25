"""IO utilities for mlip package"""
import json
import os
import pickle as pkl

import fsspec
import h5py
import torch
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from rdkit.Chem import MolFromXYZFile

gcp_filesys = fsspec.filesystem("gs")
local_filesys = LocalFileSystem()


def get_local_cache():
    cache_dir = os.path.expanduser(os.path.expandvars("~/.cache/openqdc"))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_remote_cache():
    remote_cache = "gs://opendatasets/openqdc"
    return remote_cache


def push_remote(local_path, overwrite=True):
    remote_path = local_path.replace(get_local_cache(), get_remote_cache())
    gcp_filesys.mkdirs(os.path.dirname(remote_path), exist_ok=False)
    # print(f"Pushing {local_path} file to {remote_path}, ({gcp_filesys.exists(os.path.dirname(remote_path))})")
    if not gcp_filesys.exists(remote_path) or overwrite:
        gcp_filesys.put_file(local_path, remote_path)
    return remote_path


def pull_locally(local_path, overwrite=True):
    remote_path = local_path.replace(get_local_cache(), get_remote_cache())
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path) or overwrite:
        # print(f"Pulling {remote_path} file to {local_path}")
        gcp_filesys.get_file(remote_path, local_path)
    return local_path


def copy_exists(local_path):
    remote_path = local_path.replace(get_local_cache(), get_remote_cache())
    return os.path.exists(local_path) or gcp_filesys.exists(remote_path)


def load_torch_gcs(path):
    """Loads torch file"""
    # get file system
    fs: GCSFileSystem = fsspec.filesystem("gs")

    # load from GCS
    with fs.open(path, "rb") as fp:
        return torch.load(fp)


def load_torch(path):
    """Loads torch file"""
    return torch.load(path)


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
    return MolFromXYZFile(path)
