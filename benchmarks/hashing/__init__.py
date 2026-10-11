"""The H100 hashing benchmark (HASHING_PLAN_DRAFT Section 6): CPU parts.

`run.py` is the entry point and writes one JSON report. From `verification` this package
imports only the leaf-size list (`verification.computation.leaf_sizes`), so it measures the
real step's leaf sizes without depending on the protocol's code paths.
"""
