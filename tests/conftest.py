import pytest

from kite_data import config


@pytest.fixture(autouse=True, scope="session")
def _assets():
    if not (config.BASE_URDF.exists() and config.MENAGERIE_XML.exists()):
        pytest.skip("third-party assets missing: run scripts/fetch_assets.sh")
