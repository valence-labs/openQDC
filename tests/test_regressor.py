import numpy as np
import pytest

from openqdc.datasets.potential import Dummy
from openqdc.utils.regressor import LinearSolver, Regressor, RidgeSolver


@pytest.fixture
def small_dummy():
    class SmallDummy(Dummy):
        def __len__(self):
            return 10

    return SmallDummy()


def test_small_dummy(small_dummy):
    assert len(small_dummy) == 10


def test_regressors(small_dummy):
    reg = Regressor.from_openqdc_dataset(small_dummy)
    assert isinstance(reg, Regressor)
    assert hasattr(reg, "X") and hasattr(reg, "y")
    for solvers in [("linear", LinearSolver), ("ridge", RidgeSolver)]:
        solver_type, inst = solvers[0], solvers[1]
        setattr(reg, "solver_type", solver_type)
        reg.solver = reg._get_solver()
        assert isinstance(reg.solver, inst)
        num_methods = len(small_dummy.energy_methods)
        try:
            results = reg.solve()
            assert results[0].shape[1] == num_methods
        except np.linalg.LinAlgError:
            pass
