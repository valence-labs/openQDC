import numpy as np
import pytest
from numpy import array, float32

from openqdc.datasets.potential.dummy import PredefinedDataset
from openqdc.datasets.statistics import EnergyStatistics, ForceStatistics


@pytest.fixture
def targets():
    return {
        # hartree/bohr
        "ForcesCalculatorStats": ForceStatistics(
            mean=array([[-7.4906794e-08]], dtype=float32),
            std=array([[0.02859425]], dtype=float32),
            component_mean=array([[4.6292794e-07], [-2.1531498e-07], [-4.7250555e-07]], dtype=float32),
            component_std=array([[0.02794589], [0.03237366], [0.02497733]], dtype=float32),
            component_rms=array([[0.02794588], [0.03237367], [0.02497733]], dtype=float32),
        ),
        # Hartree
        "TotalEnergyStats": EnergyStatistics(
            mean=array([[-126.0]], dtype=float32), std=array([[79.64923]], dtype=float32)
        ),
        # Hartree
        "FormationEnergyStats": EnergyStatistics(mean=array([[841.82607372]]), std=array([[448.15780975]])),
        # Hartree
        "PerAtomFormationEnergyStats": EnergyStatistics(mean=array([[20.18697415]]), std=array([[7.30153839]])),
    }


@pytest.mark.parametrize(
    "property,expected",
    [
        ("n_atoms", [27, 48, 27, 45, 45]),
        ("energies", [[-90.0], [-230.0], [-10.0], [-200.0], [-100.0]]),
    ],
)
def test_dataset_load(property, expected):
    ds = PredefinedDataset(energy_type="formation")
    assert ds is not None
    assert len(ds) == 5
    assert ds.data["atomic_inputs"].shape == (192, 5)
    assert ds.data["forces"].shape == (192, 3, 1)
    np.testing.assert_equal(ds.data[property], np.array(expected))


def test_predefined_dataset(targets):
    ds = PredefinedDataset(energy_type="formation")
    keys = ["ForcesCalculatorStats", "FormationEnergyStats", "PerAtomFormationEnergyStats", "TotalEnergyStats"]
    assert all(k in ds.get_statistics() for k in keys)
    stats = ds.get_statistics()

    formation_energy_stats = stats["FormationEnergyStats"]
    formation_energy_stats_t = targets["FormationEnergyStats"].to_dict()
    np.testing.assert_almost_equal(formation_energy_stats["mean"], formation_energy_stats_t["mean"])
    np.testing.assert_almost_equal(formation_energy_stats["std"], formation_energy_stats_t["std"])

    per_atom_formation_energy_stats = stats["PerAtomFormationEnergyStats"]
    per_atom_formation_energy_stats_t = targets["PerAtomFormationEnergyStats"].to_dict()
    np.testing.assert_almost_equal(per_atom_formation_energy_stats["mean"], per_atom_formation_energy_stats_t["mean"])
    np.testing.assert_almost_equal(per_atom_formation_energy_stats["std"], per_atom_formation_energy_stats_t["std"])

    total_energy_stats = stats["TotalEnergyStats"]
    total_energy_stats_t = targets["TotalEnergyStats"].to_dict()
    np.testing.assert_almost_equal(total_energy_stats["mean"], total_energy_stats_t["mean"])
    np.testing.assert_almost_equal(total_energy_stats["std"], total_energy_stats_t["std"])

    forces_stats = stats["ForcesCalculatorStats"]
    forces_stats_t = targets["ForcesCalculatorStats"].to_dict()
    np.testing.assert_almost_equal(forces_stats["mean"], forces_stats_t["mean"])
    np.testing.assert_almost_equal(forces_stats["std"], forces_stats_t["std"])
    np.testing.assert_almost_equal(forces_stats["component_mean"], forces_stats_t["component_mean"])
    np.testing.assert_almost_equal(forces_stats["component_std"], forces_stats_t["component_std"])
    np.testing.assert_almost_equal(forces_stats["component_rms"], forces_stats_t["component_rms"])
