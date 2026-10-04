# Proof of training steps with Freivalds checks

This repo builds a protocol that checks each training step of an LLM was computed as
claimed. It targets the same problem as PoTS (Seddik et al., arXiv:2510.15106, 2025), with
lower compute and memory cost and a real security guarantee.

## How it works

A **prover** trains the model and commits to a transcript of every step. The transcript
holds the batch, the weights, and the output of every matrix multiplication, all under one
Merkle root. A **verifier** then checks the step without rerunning it:

- Each matmul `C = A·B` is checked with Freivalds' algorithm: for a random vector `r`, it
  tests whether `A·(B·r)` equals `C·r`, within a floating-point tolerance. That costs a few
  matrix-vector products instead of a full matrix product.
- The random vectors come from the transcript's own hash (Fiat-Shamir). So the prover
  can't know them before committing, and no interaction is needed.
- Further checks tie the batch to the committed dataset and schedule, and tie each step's
  weights to the previous step's update.

PoTS is a statistical backdoor detector that retrains a tail of the model. This protocol
instead checks every matmul of every step, over the whole model.

## Status

Implementation is in progress. The work runs in two phases with one code path:

1. A **test-scale** run on CPU: SmolLM2-135M-Instruct in fp32, plain SGD, on Alpaca with a
   poisoned variant that uses the BadNets "BadMagic" trigger.
2. A **full-scale** run on GPU that repeats the PoTS experiment with 0.5B–1.5B models, so
   cost and results compare directly with the paper.

## Where things are

| Path | Contents |
|---|---|
| `docs/verification/STATUS.md` | Current stage, next task, and a map of the design docs |
| `docs/verification/VERIFICATION_PROTOCOL_SPEC.md` | The protocol spec, independent of any model architecture |
| `docs/verification/VERIFICATION_PROTOCOL_REFERENCE_BLOCK.md` | The protocol worked through on one SmolLM2 decoder step |
| `setup/` | Run setup that doesn't depend on the protocol: configuration, data, records, model loading |
| `verification/` | The protocol implementation. `verification/README.md` explains its architecture |
| `tests/` | The test suite, laid out like the code |

## Run the tests

To set up the environment, create a local virtual environment with Python 3.14 and install
the pinned versions into it. These are the versions the environment setup (task T0 of the
implementation plan) pinned, and every recorded result uses them.

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.9.1 transformers==4.57.6 datasets==5.0.0 blake3==1.0.9 \
    huggingface_hub==0.36.2 numpy==2.5.3 pytest
.venv/bin/python -m pytest tests            # fast suite
.venv/bin/python -m pytest tests -m slow    # loads the real model
```

## Origin

This repo started as a fork of
[nano-llm-posttraining](https://github.com/pochenai/nano-llm-posttraining), an educational
post-training tutorial. The protocol uses none of the tutorial code, so that code was
removed. Git history keeps it.
