import sys
import os
import pytest
from httpx import AsyncClient, ASGITransport

# Add backend and tests dirs so backend modules and test_helpers resolve
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.join(_tests_dir, "..")
sys.path.insert(0, _backend_dir)
sys.path.insert(0, _tests_dir)

from test_helpers import make_mock_rag, build_test_app


@pytest.fixture
def mock_rag():
    return make_mock_rag()


@pytest.fixture
def app(mock_rag):
    return build_test_app(mock_rag)


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
