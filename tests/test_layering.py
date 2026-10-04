"""The package layering: which top-level parts may import which.

1. `setup/` imports nothing from `verification`.
2. `verification/commitment/` and `verification/verifier/matmul_check/` import nothing from
   the prover, the runs, or the rest of the verifier.
3. `verification/verifier/` imports nothing from the prover (invariant 1).
4. `verification/computation/` imports nothing from the prover at run time, since the verifier
   builds its replay from it. The labeling code may name the capture's types under
   `TYPE_CHECKING`; nothing else in `computation/` may.

Imports under `TYPE_CHECKING` count too, except where rule 4 allows them.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _type_checking_nodes(tree: ast.AST) -> set[int]:
    """ids of the nodes inside an `if TYPE_CHECKING:` body."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) \
                and node.test.id == "TYPE_CHECKING":
            out.update(id(n) for stmt in node.body for n in ast.walk(stmt))
    return out


def _imports(path: Path, *, runtime_only: bool = False) -> set[str]:
    """Absolute names of every module `path` imports, relative imports resolved."""
    package = _module_name(path).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    out = set()
    tree = ast.parse(path.read_text())
    skip = _type_checking_nodes(tree) if runtime_only else set()
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
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


# The labeling code: it maps the prover's captured records to product slots.
LABELING = {"verification/computation/interface.py",
            "verification/computation/instances/llama.py",
            "verification/computation/instances/mlp.py"}


def test_computation_never_imports_the_prover_at_run_time():
    files = _sources("verification/computation")
    assert files
    bad = []
    for f in files:
        rel = str(f.relative_to(ROOT))
        names = _imports(f, runtime_only=rel in LABELING)
        bad += [f"{rel} imports {n}" for n in sorted(names) if _under(n, "verification.prover")]
    assert not bad, "\n".join(bad)


def test_verifier_replay_path_loads_no_prover_module():
    """The verifier and every computation instance, imported fresh, load nothing from the prover."""
    code = ("import sys, verification.verifier.driver, verification.verifier.checks, "
            "verification.computation.instances, verification.computation.substitution; "
            "print(sorted(m for m in sys.modules if m.startswith('verification.prover')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_type_checking_imports_are_found():
    f = ROOT / "verification" / "computation" / "instances" / "llama.py"
    assert "verification.prover.capture" in _imports(f)
    assert "verification.prover.capture" not in _imports(f, runtime_only=True)


def test_relative_imports_resolve():
    f = ROOT / "verification" / "computation" / "instances" / "__init__.py"
    assert "verification.computation.instances.llama" in _imports(f)
