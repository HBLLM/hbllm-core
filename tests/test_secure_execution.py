
from hbllm.actions.execution_node import validate_code


def test_secure_execution_builtins():
    bad_code = "__builtins__['__import__']('os').system('ls')"
    violations = validate_code(bad_code)
    assert len(violations) > 0
    assert any("blocked built-in access '__builtins__'" in v for v in violations)


def test_secure_execution_dunders():
    bad_code = "my_obj.__dict__['some'] = 1"
    violations = validate_code(bad_code)
    assert len(violations) > 0
    assert any("blocked dunder access '.__dict__'" in v for v in violations)


def test_secure_execution_ok():
    good_code = "a = 1 + 1\nprint(a)"
    violations = validate_code(good_code)
    assert len(violations) == 0
