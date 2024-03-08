from typing import Dict, List, Optional

import numpy as np

from openqdc.datasets.potential.base import BaseDataset
from openqdc.utils.constants import NB_ATOMIC_FEATURES


class BaseInteractionDataset(BaseDataset):
    def __init__(
        self,
        energy_unit: Optional[str] = None,
        distance_unit: Optional[str] = None,
        overwrite_local_cache: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            energy_unit=energy_unit,
            distance_unit=distance_unit,
            overwrite_local_cache=overwrite_local_cache,
            cache_dir=cache_dir,
        )

    def collate_list(self, list_entries: List[Dict]):
        # concatenate entries
        print(list_entries[0])
        res = {
            key: np.concatenate([r[key] for r in list_entries if r is not None], axis=0)
            for key in list_entries[0]
            if not isinstance(list_entries[0][key], dict)
        }

        csum = np.cumsum(res.get("n_atoms"))
        print(csum)
        x = np.zeros((csum.shape[0], 2), dtype=np.int32)
        x[1:, 0], x[:, 1] = csum[:-1], csum
        res["position_idx_range"] = x

        return res

    @property
    def data_shapes(self):
        return {
            "atomic_inputs": (-1, NB_ATOMIC_FEATURES),
            "position_idx_range": (-1, 2),
            "energies": (-1, len(self.__energy_methods__)),
            "forces": (-1, 3, len(self.force_target_names)),
        }
