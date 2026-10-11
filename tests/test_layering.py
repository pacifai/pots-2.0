"""The package layering: which top-level parts may import which.

1. `setup/` imports nothing from `verification` or `evaluation`.
2. `verification/commitment/` and `verification/verifier/matmul_check/` import nothing from
   the prover, the runs, or the rest of the verifier.
3. `verification/verifier/` imports nothing from the prover (invariant 1).
4. `verification/computation/` imports nothing from the prover at run time, since the verifier
   builds its replay from it. The labeling code may name the capture's types under
   `TYPE_CHECKING`; nothing else in `computation/` may.
5. Of this repo's packages, `evaluation/` imports `setup` only: never `verification` or `tests`.
6. `evaluation/results/` imports neither torch nor `evaluation.asr`, so a result script can't
   train, verify or score (EVALUATION_SPEC §1). A fresh interpreter that imports every result
   script and the modules they use loads no torch, `verification` or `evaluation.asr` module.
7. `verification/` imports nothing from `evaluation`, except that `verification/runs/` may
   import `evaluation.asr` (as `from evaluation.asr import ...`).

Imports under `TYPE_CHECKING` count too, except where rule 4 allows them.
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module_name(path: Path, root: Path = ROOT) -> str:
    parts = path.relative_to(root).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _type_checking_nodes(tree: ast.AST) -> set[int]:
    """ids of the nodes inside an `if TYPE_CHECKING:` body."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) \
                and node.test.id == "TYPE_CHECKING":
            out.update(id(n) for stmt in node.body for n in ast.walk(stmt))
    return out


def _imports(path: Path, *, runtime_only: bool = False, root: Path = ROOT) -> set[str]:
    """Absolute names of every module `path` imports, relative imports resolved."""
    package = _module_name(path, root).split(".")
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


def _sources(*dirs: str, root: Path = ROOT) -> list[Path]:
    return sorted(f for d in dirs for f in (root / d).rglob("*.py"))


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


class Rule(NamedTuple):
    src: str  # a directory under the repository root
    forbidden: tuple[str, ...]  # module prefixes it may not import
    allowed: tuple[str, ...] = ()  # exceptions inside `forbidden`
    skip: tuple[str, ...] = ()  # subdirectories with a rule of their own


VERIFICATION = ("verification.prover", "verification.runs", "verification.verifier")
RULES = [
    Rule("setup", ("verification", "evaluation")),
    Rule("verification/commitment", VERIFICATION),
    Rule("verification/verifier/matmul_check", VERIFICATION,
         ("verification.verifier.matmul_check",)),
    Rule("verification/verifier", ("verification.prover",)),
    Rule("evaluation", ("verification", "tests")),
    Rule("evaluation/results", ("torch", "evaluation.asr", "verification")),
    Rule("verification", ("evaluation",), skip=("verification/runs",)),
    Rule("verification/runs", ("evaluation",), ("evaluation.asr",)),
]


def _violations(rule: Rule, root: Path = ROOT) -> list[str]:
    skip = [root / d for d in rule.skip]
    files = [f for f in _sources(rule.src, root=root) if not any(f.is_relative_to(d) for d in skip)]
    assert files, rule.src
    return [f"{f.relative_to(root)} imports {name}"
            for f in files for name in sorted(_imports(f, root=root))
            if any(_under(name, p) for p in rule.forbidden)
            and not any(_under(name, p) for p in rule.allowed)]


@pytest.mark.parametrize("rule", RULES, ids=[r.src for r in RULES])
def test_layering(rule):
    bad = _violations(rule)
    assert not bad, "\n".join(bad)


def test_a_result_script_importing_verification_fails(tmp_path):
    """The rules above, applied to a copy of `evaluation/` with one bad result script."""
    for f in _sources("evaluation"):
        dst = tmp_path / f.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(f.read_text())
    bad = tmp_path / "evaluation" / "results" / "bad.py"
    bad.write_text("from verification.runs.metrics import read_records\n")
    for src in ("evaluation", "evaluation/results"):
        (rule,) = [r for r in RULES if r.src == src]
        assert _violations(rule, tmp_path) == [
            "evaluation/results/bad.py imports verification.runs.metrics",
            "evaluation/results/bad.py imports verification.runs.metrics.read_records"], src
    bad.write_text("import torch\nfrom ..asr import score\n")
    (rule,) = [r for r in RULES if r.src == "evaluation/results"]
    assert _violations(rule, tmp_path) == ["evaluation/results/bad.py imports evaluation.asr",
                                           "evaluation/results/bad.py imports evaluation.asr.score",
                                           "evaluation/results/bad.py imports torch"]


def test_result_scripts_load_no_torch_verification_or_asr():
    """Imported fresh, the result scripts and the modules they use load none of the three."""
    modules = ["evaluation.records", "evaluation.output",
               *(_module_name(f) for f in _sources("evaluation/results"))]
    code = ("import importlib, sys\n"
            f"for m in {modules!r}: importlib.import_module(m)\n"
            "print(sorted(m for m in sys.modules if m in ('torch', 'evaluation.asr')"
            " or m.split('.')[0] == 'verification'))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "[]", out.stdout


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
