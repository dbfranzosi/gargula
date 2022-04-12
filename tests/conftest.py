import pytest

'''

Every fixture is seen as a global variable accross any other
test_*.py file, for example 

The fixture input_data if declared as argument in any other function that
starts with def test_* will have the value of its return

'''

@pytest.fixture
def input_data():
    return 1