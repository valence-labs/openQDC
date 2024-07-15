import pickle as pkl
from abc import ABC, abstractmethod
from os.path import join as p_join
from typing import Callable, List, Optional

import numpy as np
import zarr

from openqdc.utils.io import pull_locally


class GeneralStructure(ABC):
    """
    Abstract Factory class for datasets type in the openQDC package.
    """

    _ext: Optional[str] = None
    _extra_files: Optional[List[str]] = None

    @property
    def ext(self):
        return self._ext

    @property
    @abstractmethod
    def load_fn(self) -> Callable:
        """
        Function to use for loading the data.
        """
        raise NotImplementedError

    def add_extension(self, filename):
        return filename + self.ext

    @abstractmethod
    def save_preprocess(self, preprocess_path, data_keys, data_dict, extra_data_keys, extra_data_types) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def load_extra_files(self, data, preprocess_path, data_keys, pkl_data_keys, overwrite):
        raise NotImplementedError

    def join_and_ext(self, path, filename):
        return p_join(path, self.add_extension(filename))

    def load_data(self, preprocess_path, data_keys, data_types, data_shapes, extra_data_keys, overwrite):
        data = {}
        for key in data_keys:
            filename = self.join_and_ext(preprocess_path, key)
            pull_locally(filename, overwrite=overwrite)
            data[key] = self.load_fn(filename, mode="r", dtype=data_types[key])
            data[key] = self.unpack(data[key])
            data[key] = data[key].reshape(*data_shapes[key])

        data = self.load_extra_files(data, preprocess_path, data_keys, extra_data_keys, overwrite)
        return data

    def unpack(self, data):
        return data


class MemMapDataset(GeneralStructure):
    """
    Dataset structure for memory-mapped numpy arrays and props.pkl files.
    """

    _ext = ".mmap"
    _extra_files = ["props.pkl"]

    @property
    def load_fn(self):
        return np.memmap

    def save_preprocess(self, preprocess_path, data_keys, data_dict, extra_data_keys, extra_data_types) -> List[str]:
        """
        Save the preprocessed data to the cache directory and optionally upload it to the remote storage.
        data_dict : dict
            Dictionary containing the preprocessed data.
        """
        local_paths = []
        for key in data_keys:
            local_path = self.join_and_ext(preprocess_path, key)
            out = np.memmap(local_path, mode="w+", dtype=data_dict[key].dtype, shape=data_dict[key].shape)
            out[:] = data_dict.pop(key)[:]
            out.flush()
            local_paths.append(local_path)

        # save smiles and subset
        local_path = p_join(preprocess_path, "props.pkl")

        # assert that (required) pkl keys are present in data_dict
        assert all([key in data_dict.keys() for key in extra_data_keys])

        # store unique and inverse indices for str-based pkl keys
        for key in extra_data_keys:
            if extra_data_types[key] == str:
                data_dict[key] = np.unique(data_dict[key], return_inverse=True)

        with open(local_path, "wb") as f:
            pkl.dump(data_dict, f)

        local_paths.append(local_path)
        return local_paths

    def load_extra_files(self, data, preprocess_path, data_keys, pkl_data_keys, overwrite):
        filename = p_join(preprocess_path, "props.pkl")
        pull_locally(filename, overwrite=overwrite)
        with open(filename, "rb") as f:
            tmp = pkl.load(f)
            all_pkl_keys = set(tmp.keys()) - set(data_keys)
            # assert required pkl_keys are present in all_pkl_keys
            assert all([key in all_pkl_keys for key in pkl_data_keys])
            for key in all_pkl_keys:
                x = tmp.pop(key)
                if len(x) == 2:
                    data[key] = x[0][x[1]]
                else:
                    data[key] = x
        return data


class ZarrDataset(GeneralStructure):
    """
    Dataset structure for zarr files.
    """

    _ext = ".zip"
    _extra_files = ["metadata.zip"]
    _zarr_version = 2

    @property
    def load_fn(self):
        return zarr.open

    def unpack(self, data):
        return data[:]

    def save_preprocess(self, preprocess_path, data_keys, data_dict, extra_data_keys, extra_data_types) -> List[str]:
        # os.makedirs(p_join(ds.root, "zips",  ds.__name__), exist_ok=True)
        local_paths = []
        for key, value in data_dict.items():
            if key not in data_keys:
                continue
            zarr_path = self.join_and_ext(preprocess_path, key)
            value = data_dict.pop(key)
            z = zarr.open(
                zarr.storage.ZipStore(zarr_path),
                "w",
                zarr_version=self._zarr_version,
                shape=value.shape,
                dtype=value.dtype,
            )
            z[:] = value[:]
            local_paths.append(zarr_path)
            # if key in attrs:
            #    z.attrs.update(attrs[key])

        metadata = p_join(preprocess_path, "metadata.zip")

        group = zarr.group(zarr.storage.ZipStore(metadata))

        for key in extra_data_keys:
            if extra_data_types[key] == str:
                data_dict[key] = np.unique(data_dict[key], return_inverse=True)

        for key, value in data_dict.items():
            # sub=group.create_group(key)
            if key in ["name", "subset"]:
                data = group.create_dataset(key, shape=value[0].shape, dtype=value[0].dtype)
                data[:] = value[0][:]
                data2 = group.create_dataset(key + "_ptr", shape=value[1].shape, dtype=np.int32)
                data2[:] = value[1][:]
            else:
                data = group.create_dataset(key, shape=value.shape, dtype=value.dtype)
                data[:] = value[:]
        local_paths.append(metadata)
        return local_paths

    def load_extra_files(self, data, preprocess_path, data_keys, pkl_data_keys, overwrite):
        filename = self.join_and_ext(preprocess_path, "metadata")
        pull_locally(filename, overwrite=overwrite)
        tmp = self.load_fn(filename)
        all_pkl_keys = set(tmp.keys()) - set(data_keys)
        # assert required pkl_keys are present in all_pkl_keys
        assert all([key in all_pkl_keys for key in pkl_data_keys])
        for key in all_pkl_keys:
            if key not in pkl_data_keys:
                data[key] = tmp[key][:][tmp[key][:]]
            else:
                data[key] = tmp[key][:]
        return data

    # TODO: checksum , maybe convert to archive instead of zips
