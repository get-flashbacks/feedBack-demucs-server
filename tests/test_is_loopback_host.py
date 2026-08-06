"""Test the loopback-host check used by main()'s unauthenticated-bind warning.

Extracts the function from server.py using AST to avoid the heavy
torch/whisperx import chain that requires GPU dependencies (same approach
as test_cache_cleanup.py's _parse_ttl extraction).

Regression coverage for a real review finding: a literal string comparison
against ("127.0.0.1", "localhost", "::1") misses equally-valid loopback
spellings — notably IPv6 loopback written without :: compression
(0:0:0:0:0:0:0:1), which would have wrongly triggered the "server is
publicly reachable and unauthenticated" warning on a host that is, in
fact, loopback-only.
"""

import ast
from pathlib import Path

import pytest

SERVER_PY = Path(__file__).parent.parent / "server.py"


def _extract_function(func_name: str):
    """Extract a function from server.py source code using AST."""
    source = SERVER_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            mod = ast.Module(body=[node], type_ignores=[])
            ast.copy_location(mod, node)
            code = compile(ast.unparse(mod), "<test>", "exec")
            namespace = {"ipaddress": __import__("ipaddress")}
            exec(code, namespace)
            return namespace[func_name]
    raise ValueError(f"Function {func_name} not found in {SERVER_PY}")


_is_loopback_host = _extract_function("_is_loopback_host")


@pytest.mark.parametrize("host", [
    "127.0.0.1",
    "localhost",
    "::1",
    "0:0:0:0:0:0:0:1",   # the exact alias a review comment flagged as broken
    "[::1]",             # bracketed IPv6 literal form
    "127.0.0.5",         # the whole 127.0.0.0/8 block is loopback, not just .1
    "127.255.255.255",
])
def test_recognizes_loopback_addresses(host):
    assert _is_loopback_host(host) is True


@pytest.mark.parametrize("host", [
    "0.0.0.0",
    "192.168.1.5",
    "nas.local",
    "",
    "10.0.0.1",
])
def test_rejects_non_loopback_addresses(host):
    assert _is_loopback_host(host) is False
