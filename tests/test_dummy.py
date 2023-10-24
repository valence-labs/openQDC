"""Path hack to make tests work."""

from openqdc.datasets.dummy import Dummy  # noqa: E402


def test_dummy():
    ds = Dummy()
    assert len(ds) > 10
    assert ds[100]
