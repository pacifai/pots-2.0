#!/usr/bin/env bash
# FS11's H100 session as one unattended job, which ends its own pod.
#
#   nohup bash verification/runs/fs11_job.sh UPLOAD_DIR OUT_DIR > /workspace/fs11_job.out 2>&1 &
#
# Run from the repo (the upload's repo/, with its .venv). In order:
#
# 1. A deadline timer starts: FS11_DEADLINE_S (default 12600, 3.5 h) after the start it
#    terminates the pod, whatever the job is doing.
# 2. The session (verification.runs.fs11_session run) prints the environment, refuses a
#    machine without an H100 (FS11_ALLOW_ANY_GPU=1 overrides), and runs its parts, each with
#    its own log under OUT_DIR. Each part's exit code is recorded and a failed part doesn't stop
#    the others. A running part is killed at the work deadline, which leaves FS11_GRACE_S plus
#    FS11_PACK_RESERVE_S before the timer fires.
# 3. Whatever the session's outcome, OUT_DIR is packed into OUT_DIR.tar, its SHA-256 printed and
#    written to OUT_DIR.tar.sha256, and OUT_DIR.READY written: the tarball can be fetched.
# 4. A grace window of FS11_GRACE_S (default 1200, 20 min) for the copy. Touching
#    OUT_DIR.FETCHED ends it early.
# 5. The pod is terminated (removed, not stopped). An exit through a signal (TERM, INT, HUP)
#    terminates it too.
#
# DRY_RUN=1 prints the terminate command instead of running it, and kills the job when the
# timer fires, as the pod's removal would. Without DRY_RUN the job refuses to start unless it
# can terminate: RUNPOD_POD_ID is set and runpodctl or RUNPOD_API_KEY is there.
#
# Runpod's terminate command, from its public docs (checked 2026-10-14):
# - https://docs.runpod.io/pods/manage-pods: "runpodctl pod delete $RUNPOD_POD_ID" terminates
#   a pod, which "permanently deletes all data not stored in a network volume";
#   "runpodctl pod stop" only stops it, and a stopped pod still bills for its disk.
# - https://docs.runpod.io/pods/references/environment-variables: RUNPOD_POD_ID is the
#   "Unique Pod identifier", RUNPOD_API_KEY the "Pod-scoped API key".
# - https://docs.runpod.io/runpodctl/overview: "Every Pod you deploy comes preinstalled with
#   runpodctl and a Pod-scoped API key" (runpodctl 2.x).
# - Fallbacks: runpodctl 1.x spells it "runpodctl remove pod <podId>"
#   (github.com/runpod/runpodctl, docs/runpodctl_remove_pod.md), and the REST API's
#   "DELETE https://rest.runpod.io/v1/pods/{podId}" with a bearer key
#   (https://docs.runpod.io/api-reference/pods/DELETE/pods/podId).
set -u

UPLOAD=${1:?usage: fs11_job.sh UPLOAD_DIR OUT_DIR}
OUT=${2:?usage: fs11_job.sh UPLOAD_DIR OUT_DIR}
DEADLINE_S=${FS11_DEADLINE_S:-12600}
GRACE_S=${FS11_GRACE_S:-1200}
PACK_RESERVE_S=${FS11_PACK_RESERVE_S:-600}
PY=${FS11_PYTHON:-.venv/bin/python}
DRY=${DRY_RUN:-0}
POD=${RUNPOD_POD_ID:-}

case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
case "$UPLOAD" in /*) ;; *) UPLOAD="$PWD/$UPLOAD" ;; esac
OUT=${OUT%/}
TAR="$OUT.tar"
READY="$OUT.READY"
FETCHED="$OUT.FETCHED"
FLAG="$OUT.TERMINATED"

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# The pod's end. Idempotent across the main job and the timer: the first caller writes FLAG.
terminate() {
  if [ -e "$FLAG" ]; then return 0; fi
  echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$FLAG"
  if [ "$DRY" = 1 ]; then
    say "DRY_RUN: terminate ($1): runpodctl pod delete ${POD:-\$RUNPOD_POD_ID}"
    say "DRY_RUN:   fallbacks: runpodctl remove pod ${POD:-\$RUNPOD_POD_ID}; curl -X DELETE" \
        "https://rest.runpod.io/v1/pods/${POD:-\$RUNPOD_POD_ID} -H 'Authorization: Bearer ...'"
    return 0
  fi
  say "terminating pod $POD ($1)"
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if command -v runpodctl >/dev/null 2>&1; then
      runpodctl pod delete "$POD" && return 0
      runpodctl remove pod "$POD" && return 0
    fi
    if [ -n "${RUNPOD_API_KEY:-}" ] && command -v curl >/dev/null 2>&1; then
      curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$POD" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" && return 0
    fi
    say "terminate attempt $i failed; retrying in 30 s"
    sleep 30
  done
  say "TERMINATE FAILED: end pod $POD by hand"
  return 1
}

# ---- before anything ---------------------------------------------------------------------
if [ "$DRY" != 1 ]; then
  if [ -z "$POD" ]; then
    say "refused: RUNPOD_POD_ID is not set, so the job could not end its pod (DRY_RUN=1 to try it)"
    exit 2
  fi
  if ! command -v runpodctl >/dev/null 2>&1 && [ -z "${RUNPOD_API_KEY:-}" ]; then
    say "refused: neither runpodctl nor RUNPOD_API_KEY is there to terminate pod $POD"
    exit 2
  fi
fi
if [ $((GRACE_S + PACK_RESERVE_S)) -ge "$DEADLINE_S" ]; then
  say "refused: FS11_GRACE_S + FS11_PACK_RESERVE_S must be below FS11_DEADLINE_S"; exit 2
fi
mkdir -p "$OUT"
rm -f "$FLAG" "$READY" "$FETCHED"
exec > >(tee -a "$OUT/job.log") 2>&1

START=$(date +%s)
HARD=$((START + DEADLINE_S))
WORK_DEADLINE=$((HARD - GRACE_S - PACK_RESERVE_S))
MAIN=$$
say "FS11 job: upload $UPLOAD, out $OUT, pod ${POD:-none}, DRY_RUN=$DRY"
say "deadline in ${DEADLINE_S} s ($(date -u -r "$HARD" +%H:%M:%SZ 2>/dev/null || date -u -d "@$HARD" +%H:%M:%SZ)); parts stop by $((WORK_DEADLINE - START)) s; grace ${GRACE_S} s"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_TOKEN_PATH
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi

# ---- the deadline timer ------------------------------------------------------------------
(
  trap 'kill $SLEEP 2>/dev/null; exit 0' TERM
  sleep "$DEADLINE_S" & SLEEP=$!
  wait $SLEEP
  say "deadline reached after ${DEADLINE_S} s"
  terminate "deadline"
  # In a dry run, stand in for the pod's removal: end the job (its trap stops the session).
  if [ "$DRY" = 1 ]; then kill -TERM "$MAIN" 2>/dev/null; fi
) &
TIMER=$!

on_signal() {
  say "job stopped by a signal"
  if [ -n "${ORCH:-}" ]; then
    pkill -TERM -P "$ORCH" 2>/dev/null
    kill -TERM "$ORCH" 2>/dev/null
  fi
  terminate "signal"
  kill "$TIMER" 2>/dev/null
  exit 1
}
trap on_signal TERM INT HUP

# ---- the session -------------------------------------------------------------------------
say "session: $PY -m verification.runs.fs11_session run"
"$PY" -m verification.runs.fs11_session run --upload "$UPLOAD" --out "$OUT" \
  --deadline "$WORK_DEADLINE" ${FS11_PLAN:+--plan "$FS11_PLAN"} &
ORCH=$!
wait $ORCH
RC=$?
ORCH=
say "session exit $RC"
echo "$RC" > "$OUT/session_exit_code"

# ---- pack, publish, grace, terminate -----------------------------------------------------
say "packing $OUT"
tar -cf "$TAR.tmp" -C "$(dirname "$OUT")" "$(basename "$OUT")" && mv "$TAR.tmp" "$TAR"
SUM=$(sha256 "$TAR")
echo "$SUM  $(basename "$TAR")" > "$TAR.sha256"
NOW=$(date +%s)
END=$((NOW + GRACE_S))
[ "$END" -gt "$HARD" ] && END=$HARD
{
  echo "tarball=$TAR"
  echo "sha256=$SUM"
  echo "bytes=$(wc -c < "$TAR" | tr -d ' ')"
  echo "session_exit=$RC"
  echo "terminate_at_epoch=$END"
} > "$READY"
say "READY: $TAR sha256 $SUM; terminate in $((END - NOW)) s (touch $FETCHED to end sooner)"
while [ ! -e "$FETCHED" ]; do
  LEFT=$((END - $(date +%s)))
  [ "$LEFT" -le 0 ] && break
  sleep $((LEFT < 5 ? LEFT : 5))
done
[ -e "$FETCHED" ] && say "fetch confirmed"
terminate "end of job (session exit $RC)"
kill "$TIMER" 2>/dev/null
pkill -P "$TIMER" 2>/dev/null
exit "$RC"
