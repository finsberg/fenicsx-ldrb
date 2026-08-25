import io4dolfinx
import pytest


@pytest.fixture(scope="session", autouse=True)
def global_setup_and_teardown():
    io4dolfinx.set_default_backend("h5py")
    yield
