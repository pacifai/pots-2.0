"""One script per result of EVALUATION_SPEC §5 and §7.

A script reads records through ``evaluation.records``, writes through ``evaluation.output``, and
imports neither torch, ``setup`` nor ``evaluation.asr`` (``tests/test_layering.py``).
"""
