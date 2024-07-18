from os.path import join as p_join
from pathlib import Path

import pytest

from openqdc.datasets import QM7


@pytest.mark.download
def test_API_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds = QM7(cache_dir=tmp_path)
    for filename in ["energies.mmap", "position_idx_range.mmap", "atomic_inputs.mmap", "props.pkl"]:
        assert (Path(p_join(tmp_path, ds.preprocess_path, filename))).exists()
    monkeypatch.undo()
