import time

import pytest


@pytest.mark.parametrize("i", range(20))
def test_dummy(i):
    time.sleep(4)
