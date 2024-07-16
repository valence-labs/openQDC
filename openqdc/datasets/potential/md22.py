from os.path import join as p_join

import numpy as np

from openqdc.datasets.potential.revmd17 import RevMD17, shape_atom_inputs

trajectories = [
    "Ac-Ala3-NHMe",
    "DHA",
    "stachyose",
    "AT-AT",
    "AT-AT-CG-CG",
    "double-walled_nanotube",
    "buckyball-catcher",
]


def read_npz_entry(filename, root):
    data = np.load(create_path(filename, root))
    nuclear_charges, coords, energies, forces = (
        data["z"],
        data["R"],
        data["E"],
        data["F"],
    )
    frames = coords.shape[0]
    res = dict(
        name=np.array([filename] * frames),
        subset=np.array([filename] * frames),
        energies=energies.reshape(-1, 1).astype(np.float64),
        forces=forces.reshape(-1, 3, 1).astype(np.float32),
        atomic_inputs=shape_atom_inputs(coords, nuclear_charges),
        n_atoms=np.array([len(nuclear_charges)] * frames, dtype=np.int32),
    )
    return res


def create_path(filename, root):
    return p_join(root, filename + ".npz")


class MD22(RevMD17):
    """
    MD22 consists of molecular dynamics (MD) trajectories of four major classes of biomolecules and supramolecules,
    ranging from a small peptide with 42 atoms to a double-walled nanotube with 370 atoms. The simulation trajectories
    are sampled at 400K and 500K with a resolution of 1fs. Potential energy and forces are computed using the PBE+MBD
    level of theory.

    Usage:
    ```python
    from openqdc.datasets import MD22
    dataset = MD22()
    ```

    Reference:
        https://arxiv.org/abs/2209.14865
    """

    __name__ = "md22"
    __links__ = {
        f"{x}.npz": f"http://www.quantum-machine.org/gdml/repo/datasets/md22_{x}.npz"
        for x in [
            "Ac-Ala3-NHMe",
            "DHA",
            "stachyose",
            "AT-AT",
            "AT-AT-CG-CG",
            "double-walled_nanotube",
            "buckyball-catcher",
        ]
    }

    def read_raw_entries(self):
        entries_list = []
        for trajectory in trajectories:
            entries_list.append(read_npz_entry(trajectory, self.root))
        return entries_list
