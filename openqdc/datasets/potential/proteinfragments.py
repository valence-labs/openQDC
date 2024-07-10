import os
from os.path import join as p_join

import numpy as np
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils.io import get_local_cache
from openqdc.utils.package_utils import requires_package


def convert_entries(r, e, f, z, subset):
    coordinates = r
    species = z
    forces = f
    energies = e
    n_atoms = coordinates.shape[0]
    flattened_coordinates = coordinates[:].reshape((-1, 3))
    xs = np.stack((species[:].flatten(), np.zeros(flattened_coordinates.shape[0])), axis=-1)
    res = dict(
        name=np.array([subset]),
        subset=np.array([subset]),
        energies=energies[:].reshape((-1, 1)).astype(np.float64),
        atomic_inputs=np.concatenate((xs, flattened_coordinates), axis=-1, dtype=np.float32),
        n_atoms=np.array([n_atoms], dtype=np.int32),
        forces=forces[:].reshape(-1, 3, 1).astype(np.float32),
    )
    return res


@requires_package("apsw")
def read_db(path):
    database = Database(path)
    subset = os.path.basename(path).split(".")[0]
    # Read an entry from the database.
    # entry = 0
    n = len(database)
    entries = []
    for entry in tqdm(range(n)):
        q, s, z, r, e, f, d = database[entry]
        entries.append(convert_entries(r, e, f, z, subset))
    return entries

    # assert entry < len(database)
    # q, s, z, r, e, f, d = database[entry]
    # with np.printoptions(threshold=7):
    #  print(f'entry {entry} of {len(database)}')
    #  print('total charge\n', q)
    #  print('number of unpaired electrons\n', s)
    #  print('atomic numbers\n', z)
    #  print('positions [Å]\n', r)
    #  print('energy [eV]\n', e)
    #  print('forces [eV/Å]\n', f)
    #  print('dipole [e*Å]\n', d)


class Database:
    @requires_package("apsw")
    def __init__(self, filename):
        import apsw

        self.cursor = apsw.Connection(filename, flags=apsw.SQLITE_OPEN_READONLY).cursor()

    def __len__(self):
        return self.cursor.execute("""SELECT * FROM metadata WHERE id=1""").fetchone()[-1]

    def __getitem__(self, idx):
        data = self.cursor.execute("""SELECT * FROM data WHERE id=""" + str(idx)).fetchone()
        return self._unpack_data_tuple(data)

    def _deblob(self, buffer, dtype, shape=None):
        array = np.frombuffer(buffer, dtype)
        if not np.little_endian:
            array = array.byteswap()
        array.shape = shape
        return np.copy(array)

    def _unpack_data_tuple(self, data):
        n = len(data[3]) // 4  # A single int32 is 4 bytes long.
        q = np.asarray([0.0 if data[1] is None else data[1]], dtype=np.float32)
        s = np.asarray([0.0 if data[2] is None else data[2]], dtype=np.float32)
        z = self._deblob(data[3], dtype=np.int32, shape=(n,))
        r = self._deblob(data[4], dtype=np.float32, shape=(n, 3))
        e = np.asarray([0.0 if data[5] is None else data[5]], dtype=np.float32)
        f = self._deblob(data[6], dtype=np.float32, shape=(n, 3))
        d = self._deblob(data[7], dtype=np.float32, shape=(1, 3))
        return q, s, z, r, e, f, d


# graphs is smiles
class ProteinFragments(BaseDataset):
    """https://www.science.org/doi/10.1126/sciadv.adn4397"""

    __name__ = "proteinfragments"

    __energy_methods__ = [
        PotentialMethod.WB97X_6_31G_D,  # "wb97x/6-31g(d)"
    ]

    energy_target_names = [
        "ωB97x:6-31G(d) Energy",
    ]
    # PBE0/def2-TZVPP+MBD
    __energy_unit__ = "ev"
    __distance_unit__ = "ang"
    __forces_unit__ = "ev/ang"
    __links__ = {
        f"{name}.db": f"https://zenodo.org/records/10720941/files/{name}.db?download=1"
        for name in ["general_protein_fragments"]
    }

    @property
    def root(self):
        return p_join(get_local_cache(), "proteinfragments")

    @property
    def config(self):
        assert len(self.__links__) > 0, "No links provided for fetching"
        return dict(dataset_name="proteinfragments", links=self.__links__)

    @property
    def preprocess_path(self):
        path = p_join(self.root, "preprocessed", self.__name__)
        os.makedirs(path, exist_ok=True)
        return path

    def read_raw_entries(self):
        samples = []
        for name in self.__links__:
            raw_path = p_join(self.root, f"{name}")
            samples.extend(read_db(raw_path))
        return samples


class MDDataset(ProteinFragments):
    """
    Part of the proteinfragments dataset that is generated from the molecular dynamics with their model.
    """

    __name__ = "mddataset"

    __links__ = {
        f"{name}.db": f"https://zenodo.org/records/10720941/files/{name}.db?download=1"
        for name in ["acala15nme_folding_clusters", "crambin", "minimahopping_acala15lysh", "minimahopping_acala15nme"]
    }
