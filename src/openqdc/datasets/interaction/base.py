from typing import Dict, List, Optional, Union
from openqdc.utils.io import (
    copy_exists,
    dict_to_atoms,
    get_local_cache,
    load_hdf5_file,
    load_pkl,
    pull_locally,
    push_remote,
    set_cache_dir,
)
from openqdc.datasets.potential.base import BaseDataset

from loguru import logger

import numpy as np

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
            cache_dir=cache_dir
        )

    def collate_list(self, list_entries: List[Dict]):
        # concatenate entries
        res = {key: np.concatenate([r[key] for r in list_entries if r is not None], axis=0) \
               for key in list_entries[0] if not isinstance(list_entries[0][key], dict)}

        csum = np.cumsum(res.get("n_atoms"))
        x = np.zeros((csum.shape[0], 2), dtype=np.int32)
        x[1:, 0], x[:, 1] = csum[:-1], csum
        res["position_idx_range"] = x

        return res
