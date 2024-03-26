try:
    from openqdc.datasets.interaction import DummyInteraction

    dummy_loaded = True
except:  # noqa
    dummy_loaded = False


def test_import():
    assert dummy_loaded


def test_init():
    DummyInteraction()


def test_len():
    ds = DummyInteraction()
    assert len(ds) == 9999
