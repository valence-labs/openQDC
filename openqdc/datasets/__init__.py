from .interaction import *
from .potential import *

AVAILABLE_DATASETS = {**AVAILABLE_POTENTIAL_DATASETS, **AVAILABLE_INTERACTION_DATASETS}


def _level_of_theory_overlap(dataset_collection):
    import itertools
    from itertools import groupby

    dataset_map = {}
    for dataset in dataset_collection:
        dataset_map[dataset.lower().replace("_", "")] = dataset_collection[dataset].no_init().energy_methods

    common_values_dict = {}

    for key, values in dataset_map.items():
        for value in values:
            if value in common_values_dict:
                common_values_dict[value].append(key)
            else:
                common_values_dict[value] = [key]

    return dict(filter(lambda x: len(x[1]) > 1, common_values_dict.items()))


COMMON_MAP_POTENTIALS = _level_of_theory_overlap(AVAILABLE_POTENTIAL_DATASETS)
COMMON_MAP_INTERACTIONS = _level_of_theory_overlap(AVAILABLE_INTERACTION_DATASETS)
