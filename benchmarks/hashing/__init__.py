"""The H100 hashing benchmark (HASHING_PLAN_DRAFT Section 6): CPU parts.

`run.py` is the entry point and writes one JSON report. From `verification` this package
imports only the leaf-size list (`verification.computation.leaf_sizes`), so it measures the
real step's leaf sizes without calling the protocol's hashing or challenge code. Building that
list constructs a `DecoderComputation`, but nothing here hashes, commits or draws a challenge
through `verification`: the trees, the finiteness check and the challenge path are reproduced
here and tested against the protocol's.
"""
