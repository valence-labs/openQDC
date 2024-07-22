import os
from functools import partial
from os.path import join as p_join

import datamol as dm
import numpy as np
import pandas as pd

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils.io import get_local_cache

# fmt: off
FILE_NUM = [
    "43005175","43005205","43005208","43005211","43005214","43005223",
    "43005235","43005241","43005244","43005247","43005253","43005259",
    "43005265","43005268","43005271","43005274","43005277","43005280",
    "43005286","43005292","43005298","43005304","43005307","43005313",
    "43005319","43005322","43005325","43005331","43005337","43005343"
    "43005346","43005349","43005352","43005355","43005358","43005364",
    "43005370","43005406","43005409","43005415","43005418","43005421",
    "43005424","43005427","43005430","43005433","43005436","43005439",
    "43005442","43005448","43005454","43005457","43005460","43005463",
    "43005466","43005469","43005472","43005475","43005478","43005481",
    "43005484","43005487","43005490","43005496","43005499","43005502",
    "43005505","43005508","43005511","43005514","43005517","43005520",
    "43005523","43005526","43005532","43005538","43005544","43005547",
    "43005550","43005553","43005556","43005559","43005562","43005577",
    "43005580","43005583","43005589","43005592","43005598","43005601",
    "43005616","43005622","43005625","43005628","43005634","43005637",
    "43005646","43005649","43005658","43005661","43005676","43006159",
    "43006162","43006165","43006168","43006171","43006174","43006177",
    "43006180","43006186","43006207","43006210","43006213","43006219",
    "43006222","43006228","43006231","43006273","43006276","43006279",
    "43006282","43006288","43006294","43006303","43006318","43006324",
    "43006330","43006333","43006336","43006345","43006354","43006372",
    "43006381","43006384","43006390","43006396","43006405","43006408",
    "43006411","43006423","43006432","43006456","43006468","43006471",
    "43006477","43006486","43006489","43006492","43006498","43006501",
    "43006513","43006516","43006522","43006525","43006528","43006531",
    "43006534","43006537","43006543","43006546","43006576","43006579",
    "43006603","43006609","43006615","43006621","43006624","43006627",
    "43006630","43006633","43006639","43006645","43006651","43006654",
    "43006660","43006663","43006666","43006669","43006672","43006681",
    "43006690","43006696","43006699","43006711","43006717","43006738",
    "43006747","43006756","43006762","43006765","43006768","43006771",
    "43006777","43006780","43006786","43006789","43006795","43006798",
    "43006801","43006804","43006816","43006822","43006837","43006840",
    "43006846","43006855","43006858","43006861","43006864","43006867",
    "43006870","43006873","43006876","43006882","43006897","43006900",
    "43006903","43006909","43006912","43006927","43006930","43006933",
    "43006939","43006942","43006948","43006951","43006954","43006957",
    "43006966","43006969","43006978","43006984","43006993","43006996",
    "43006999","43007002","43007005","43007008","43007011","43007014",
    "43007017","43007032","43007035","43007041","43007044","43007047",
    "43007050","43007053","43007056","43007062","43007068","43007080",
    "43007098","43007110","43007119","43007122","43007125",
]
# fmt: on


def extract_from_row(row, file_idx=None):
    smiles = row["smile"]
    z = np.array(row["z"])[:, None]
    c = np.zeros_like(z)
    x = np.concatenate((z, c), axis=1)
    positions = np.array(row["pos"]).reshape(-1, 3)

    res = dict(
        name=np.array([smiles]),
        subset=np.array(["qm1b"]) if file_idx is None else np.array([f"qm1b_{file_idx}"]),
        energies=np.array([row["energy"]]).astype(np.float64)[:, None],
        atomic_inputs=np.concatenate((x, positions), axis=-1, dtype=np.float32),
        n_atoms=np.array([x.shape[0]], dtype=np.int32),
    )
    return res


class QM1B(BaseDataset):
    """
    QM1B is a dataset containing 1 billion conformations for 1.09M small molecules generated using a custom
    PySCF library that incorporates hardware acceleration via IPUs. The molecules contain 9-11 heavy atoms and are
    subsampled from the Generated Data Bank (GDB). For each molecule, 1000 geometries are generated using RDKit.
    Electronic properties for each conformation are then calculated using the density functional B3LYP
    and the basis set STO-3G.

    Usage:
    ```python
    from openqdc.datasets import QM1B
    dataset = QM1B()
    ```

    References:
        https://arxiv.org/pdf/2311.01135\n
        https://github.com/graphcore-research/qm1b-dataset/
    """

    __name__ = "qm1b"

    __energy_methods__ = [PotentialMethod.B3LYP_STO3G]
    __force_methods__ = []

    energy_target_names = ["b3lyp/sto-3g"]
    force_target_names = []

    __energy_unit__ = "ev"
    __distance_unit__ = "bohr"
    __forces_unit__ = "ev/bohr"
    __links__ = {
        "qm1b_validation.parquet": "https://ndownloader.figshare.com/files/43005175",
        **{f"part_{i:03d}.parquet": f"https://ndownloader.figshare.com/files/{FILE_NUM[i]}" for i in range(0, 256)},
    }

    @property
    def root(self):
        return p_join(get_local_cache(), "qm1b")

    @property
    def preprocess_path(self):
        path = p_join(self.root, "preprocessed", self.__name__)
        os.makedirs(path, exist_ok=True)
        return path

    def read_raw_entries(self):
        filenames = list(map(lambda x: p_join(self.root, f"part_{x:03d}.parquet"), list(range(0, 256)))) + [
            p_join(self.root, "qm1b_validation.parquet")
        ]

        def read_entries_parallel(filename):
            df = pd.read_parquet(filename)

            def extract_parallel(df, i):
                return extract_from_row(df.iloc[i])

            fn = partial(extract_parallel, df)
            list_of_idxs = list(range(len(df)))
            results = dm.utils.parallelized(fn, list_of_idxs, scheduler="threads", progress=False)
            return results

        list_of_list = dm.utils.parallelized(read_entries_parallel, filenames, scheduler="processes", progress=True)

        return [x for xs in list_of_list for x in xs]


class QM1B_SMALL(QM1B):
    """
    QM1B_SMALL is a subset of the QM1B dataset containing a maximum of 15 random conformers per molecule.

    Usage:
    ```python
    from openqdc.datasets import QM1B_SMALL
    dataset = QM1B_SMALL()
    ```
    """

    __name__ = "qm1b_small"
