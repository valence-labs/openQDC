_DATALOADER_PACKAGE = {
    0: "numpy",
    1: "torch",
    2: "jax",
}

_VERSION = 0
try:
    import torch
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader

    _VERSION = 1

    def convert_array(data, dtype, default=None):
        """Converts numpy array to tensor of specified type"""
        if data is not None:
            return torch.tensor(data, dtype=dtype) if not isinstance(data, torch.Tensor) else data.type(dtype)

        return default

except ImportError:
    import jax
    import torch

    _VERSION = 2


class LoaderDispatch:
    def __init__(self, version):
        self.version = version

    def __call__(self, data, batch_size, **kwargs):
        if self.version == 0:
            return TorchDataLoader(data, batch_size, **kwargs)
        elif self.version == 1:
            return JaxDataLoader(data, batch_size, **kwargs)
        else:
            raise NotImplementedError("No dataloader available for this version")


class LoaderMixin:
    # def as_dataloader(self, batch_size=16):
    #    return LoaderDispatch(_VERSION)(self, batch_size)

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError("This method must be implemented")


class TorchMixin(BunchMixin):
    # def as_dataloader(self, batch_size=16):
    #    return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def convert(self, data):
        # convert to tensors
        formation = convert_array(data.energies - data.e0.sum(), torch.float32)
        positions = convert_array(data.positions, torch.float32)
        atomic_numbers = convert_array(data.atomic_numbers, torch.long)
        charges = convert_array(data.charges, torch.float32)
        forces = convert_array(data.forces, torch.float32)
        energy = convert_array(data.energies, torch.float32)
        e0 = convert_array(data.e0, torch.float32)
        num_nodes = positions.shape[0]

    # positions=positions,
    # atomic_numbers=z,
    # charges=c,
    # e0=self.__isolated_atom_energies__[..., z, c + shift].T,
    # linear_e0=self.new_e0s[..., z, c + shift].T if hasattr(self, "new_e0s") else None,
    # energies=energies,
    # name=name,
    # subset=subset,
    # forces=forces,

    def __getitem__(self, idx):
        return Data.from_dict(
            {
                "positions": positions,
                "atomic_numbers": atomic_numbers,
                "charges": charges,
                "energy": energy,
                "e0": e0,
                "forces": forces,
                "num_nodes": num_nodes,
                "formation": formation,
                "idx": idx,
            }
        )


class BunchMixin(LoaderMixin):
    def __getitem__(self, idx):
        shift = IsolatedAtomEnergyFactory.max_charge
        p_start, p_end = self.data["position_idx_range"][idx]
        input = self.data["atomic_inputs"][p_start:p_end]
        z, c, positions, energies = (
            np.array(input[:, 0], dtype=np.int32),
            np.array(input[:, 1], dtype=np.int32),
            np.array(input[:, -3:], dtype=np.float32),
            np.array(self.data["energies"][idx], dtype=np.float32),
        )
        name = self.__smiles_converter__(self.data["name"][idx])
        subset = self.data["subset"][idx]

        if "forces" in self.data:
            forces = np.array(self.data["forces"][p_start:p_end], dtype=np.float32)
        else:
            forces = None
        return Bunch(
            positions=positions,
            atomic_numbers=z,
            charges=c,
            e0=self.__isolated_atom_energies__[..., z, c + shift].T,
            linear_e0=self.new_e0s[..., z, c + shift].T if hasattr(self, "new_e0s") else None,
            energies=energies,
            name=name,
            subset=subset,
            forces=forces,
        )


def convert(data, idx):
    def convert_to_tensor(data, dtype, default=None):
        """Converts numpy array to tensor of specified type"""
        if data is not None:
            return torch.tensor(data, dtype=dtype) if not isinstance(data, torch.Tensor) else data.type(dtype)

        return default

    # convert to tensors
    formation = convert_to_tensor(data.energies - data.e0.sum(), torch.float32)
    positions = convert_to_tensor(data.positions, torch.float32)
    atomic_numbers = convert_to_tensor(data.atomic_numbers, torch.long)
    charges = convert_to_tensor(data.charges, torch.float32)
    forces = convert_to_tensor(data.forces, torch.float32)
    energy = convert_to_tensor(data.energies, torch.float32)
    e0 = convert_to_tensor(data.e0, torch.float32)
    num_nodes = positions.shape[0]

    return Data.from_dict(
        {
            "positions": positions,
            "atomic_numbers": atomic_numbers,
            "charges": charges,
            "energy": energy,
            "e0": e0,
            "forces": forces,
            "num_nodes": num_nodes,
            "formation": formation,
            "idx": idx,
        }
    )
