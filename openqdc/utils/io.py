"""IO utilities for mlip package"""
import json
import os
import pickle

import fsspec
import h5py
import torch
from gcsfs import GCSFileSystem


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
    """Saves pickle file"""
    print(f"Saving file at {path}")
    with fsspec.open(path, "wb") as fp:  # Pickling
        pickle.dump(file, fp)
    print("Done")


def load_pkl_gcs(path, check=True):
    """Load pickle file from GCS FileSystem"""
    if check:
        if not check_file_gcs(path):
            raise FileNotFoundError(f"File {path} does not exist on GCS and local.")

    # get file system
    fs: GCSFileSystem = fsspec.filesystem("gs")

    with fs.open(path, "rb") as fp:  # Unpickling
        return pickle.load(fp)


def load_pkl(path, check=True):
    """Load pickle file"""
    if check:
        if not check_file(path):
            raise FileNotFoundError(f"File {path} does not exist on GCS and local.")

    with open(path, "rb") as fp:  # Unpickling
        return pickle.load(fp)


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
