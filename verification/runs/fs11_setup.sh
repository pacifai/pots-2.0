#!/bin/bash
# The pod's environment in one command: the Mac's exact Python in a venv, and the packages the
# upload's requirements file lists.
#
#   bash UPLOAD_DIR/repo/verification/runs/fs11_setup.sh UPLOAD_DIR
#
# UPLOAD_DIR is the extracted upload (it holds manifest.json, requirements-fs11.txt and repo/).
# fs11_prep records the packing machine's Python version in manifest.json, and fs11_session
# refuses to run on any other: CPython 3.14.0 can hang at exit (the resource tracker's
# deadlock, gh-146313; 3.14.7 does not). The image's uv 0.9.0 offers no 3.14.7, so this
# installs a current uv into /root/uvnew with the system python3's pip, then
#   uv python install VERSION, uv venv --python VERSION /root/venv,
# links UPLOAD_DIR/repo/.venv to the venv, and installs the requirements. It prints the venv's
# Python and torch versions. Safe to rerun: each step replaces what an earlier run left.
#
# Names for the tests (the defaults are the pod's): FS11_SETUP_PYTHON3 (/usr/bin/python3),
# FS11_SETUP_UVDIR (/root/uvnew), FS11_SETUP_UV (UVDIR/bin/uv), FS11_SETUP_VENV (/root/venv).
set -euo pipefail

fail() { echo "fs11_setup: $*" >&2; exit 1; }

[ $# -eq 1 ] || fail "usage: bash fs11_setup.sh UPLOAD_DIR"
UPLOAD=$1
PY3=${FS11_SETUP_PYTHON3:-/usr/bin/python3}
UVDIR=${FS11_SETUP_UVDIR:-/root/uvnew}
UV=${FS11_SETUP_UV:-$UVDIR/bin/uv}
VENV=${FS11_SETUP_VENV:-/root/venv}

[ -d "$UPLOAD" ] || fail "$UPLOAD is not a directory"
[ -f "$UPLOAD/manifest.json" ] || fail "$UPLOAD/manifest.json is missing (the extracted upload?)"
[ -f "$UPLOAD/requirements-fs11.txt" ] || fail "$UPLOAD/requirements-fs11.txt is missing"
[ -d "$UPLOAD/repo" ] || fail "$UPLOAD/repo is missing"

VERSION=$("$PY3" -c '
import json, sys
try:
    v = json.load(open(sys.argv[1])).get("python_version")
except (OSError, ValueError, AttributeError) as e:
    sys.exit("manifest.json is unreadable: %s: %s" % (type(e).__name__, e))
if not v or not isinstance(v, str):
    sys.exit("manifest.json records no python_version (an older upload): repack it with fs11_prep")
print(v)
' "$UPLOAD/manifest.json") || fail "cannot read the Mac's Python version from the manifest"
echo "fs11_setup: the Mac's Python is $VERSION"

if [ -z "${FS11_SETUP_UV:-}" ]; then
    echo "fs11_setup: installing a current uv into $UVDIR"
    "$PY3" -m pip install -q --target "$UVDIR" uv || fail "pip could not install uv into $UVDIR"
fi
[ -x "$UV" ] || fail "no uv at $UV"

"$UV" python install "$VERSION" || fail "uv could not install Python $VERSION"
"$UV" venv --clear --python "$VERSION" "$VENV" || fail "uv could not make the venv at $VENV"
ln -sfn "$VENV" "$UPLOAD/repo/.venv" || fail "cannot link $UPLOAD/repo/.venv to $VENV"
"$UV" pip install --python "$VENV/bin/python" -r "$UPLOAD/requirements-fs11.txt" \
    || fail "uv could not install $UPLOAD/requirements-fs11.txt"

"$VENV/bin/python" -c '
import platform, torch
print("fs11_setup: venv Python", platform.python_version(), "torch", torch.__version__)
' || fail "the venv's Python cannot import torch"
