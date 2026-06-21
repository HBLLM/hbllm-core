"""Tests for reflex library — system, environment, routine, security."""

from hbllm.brain.autonomy.reflexes import get_all_reflexes
from hbllm.brain.autonomy.reflexes.environment import get_environment_reflexes
from hbllm.brain.autonomy.reflexes.routine import get_routine_reflexes
from hbllm.brain.autonomy.reflexes.security import get_security_reflexes
from hbllm.brain.autonomy.reflexes.system import get_system_reflexes


class TestReflexLibrary:
    """Tests for the reflex library aggregator.

    Each get_*_reflexes() returns a dict[str, Callable] mapping
    reflex names to their handler functions.
    """

    def test_all_reflexes_loaded(self):
        """get_all_reflexes returns 23 reflexes."""
        reflexes = get_all_reflexes()
        assert len(reflexes) == 23

    def test_system_reflexes_count(self):
        """System module has 7 reflexes."""
        assert len(get_system_reflexes()) == 7

    def test_environment_reflexes_count(self):
        """Environment module has 8 reflexes."""
        assert len(get_environment_reflexes()) == 8

    def test_routine_reflexes_count(self):
        """Routine module has 4 reflexes."""
        assert len(get_routine_reflexes()) == 4

    def test_security_reflexes_count(self):
        """Security module has 4 reflexes."""
        assert len(get_security_reflexes()) == 4

    def test_all_reflexes_have_names(self):
        """Every reflex has a non-empty name (dict key)."""
        for name in get_all_reflexes():
            assert name, "Empty reflex name"
            assert isinstance(name, str)

    def test_all_reflexes_are_callable(self):
        """Every reflex value is a callable function."""
        for name, handler in get_all_reflexes().items():
            assert callable(handler), f"Reflex {name} handler is not callable"

    def test_unique_names(self):
        """All reflex names are unique (dicts guarantee this by default)."""
        reflexes = get_all_reflexes()
        assert isinstance(reflexes, dict)
        # Cross-module uniqueness: check total vs sum of parts
        total = (
            len(get_system_reflexes())
            + len(get_environment_reflexes())
            + len(get_routine_reflexes())
            + len(get_security_reflexes())
        )
        assert len(reflexes) == total

    def test_system_reflex_battery_critical(self):
        """Battery critical reflex exists."""
        reflexes = get_system_reflexes()
        battery_reflexes = [n for n in reflexes if "battery" in n.lower()]
        assert len(battery_reflexes) >= 1

    def test_environment_reflex_smoke(self):
        """Smoke detection reflex exists."""
        reflexes = get_environment_reflexes()
        smoke = [n for n in reflexes if "smoke" in n.lower() or "co" in n.lower()]
        assert len(smoke) >= 1

    def test_security_reflex_login(self):
        """Unusual login reflex exists."""
        reflexes = get_security_reflexes()
        login = [n for n in reflexes if "login" in n.lower()]
        assert len(login) >= 1
