import pytest


def pytest_addoption(parser):
    parser.addoption("--record", action="store_true",
                     help="rewrite tests/golden.json from the current code")


@pytest.fixture
def record(request):
    return request.config.getoption("--record")
