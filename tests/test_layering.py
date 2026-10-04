"""The package layering: which top-level parts may import which.

1. `setup/` imports nothing from `verification`.
2. `verification/commitment/` and `verification/verifier/matmul_check/` import nothing from
   the prover, the runs, or the rest of the verifier.
3. `verification/verifier/` imports nothing from the prover (invariant 1).

Imports under `TYPE_CHECKING` count too.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: Path) -> set[str]:
    """Absolute names of every module `path` imports, relative imports resolved."""
    package = _module_name(path).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            mod = ".".join([*base, *(node.module or "").split(".")]).strip(".")
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
    return out


def _sources(*dirs: str) -> list[Path]:
    return sorted(f for d in dirs for f in (ROOT / d).rglob("*.py"))


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


RULES = [
    ("setup", ["verification"], []),
    ("verification/commitment",
     ["verification.prover", "verification.runs", "verification.verifier"], []),
    ("verification/verifier/matmul_check",
     ["verification.prover", "verification.runs", "verification.verifier"],
     ["verification.verifier.matmul_check"]),
    ("verification/verifier", ["verification.prover"], []),
]


@pytest.mark.parametrize("src,forbidden,allowed", RULES, ids=[r[0] for r in RULES])
def test_layering(src, forbidden, allowed):
    files = _sources(src)
    assert files, src
    bad = [f"{f.relative_to(ROOT)} imports {name}"
           for f in files for name in sorted(_imports(f))
           if any(_under(name, p) for p in forbidden)
           and not any(_under(name, p) for p in allowed)]
    assert not bad, "\n".join(bad)


def test_relative_imports_resolve():
    f = ROOT / "verification" / "computation" / "instances" / "__init__.py"
    assert "verification.computation.instances.llama" in _imports(f)
