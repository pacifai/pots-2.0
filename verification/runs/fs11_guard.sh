#!/bin/sh
# FS11's standalone billing guard: SECONDS from now (default 16200, 4.5 h) it removes this pod,
# whatever else runs or hangs. The boot guard in the pod's start command (verification/
# README.md, "Workflow: the first H100 session") arms the same thing at creation; this script
# re-arms it by hand, for instance after a pod restart, which ends the boot guard.
#
#   sh verification/runs/fs11_guard.sh [SECONDS]
#
# It detaches at once (setsid where there is one, nohup, stdin from /dev/null) and logs to
# FS11_GUARD_LOG (default /dev/null). When it fires it tries runpodctl pod delete, runpodctl
# 1.x's remove pod and the REST DELETE (the key on curl's stdin) in three rounds 30 s apart,
# each call under a 60 s limit; if none succeeds, runpodctl pod stop, three times. It never
# checks that the pod went: the job does that (fs11_job.sh, whose header cites Runpod's docs
# for each command). Arming it twice leaves two guards; the earlier one fires first. It prints
# the guard's pid, which `kill` disarms.
#
# Without the repo on the pod, this line does the core of it:
#
#   nohup setsid sh -c 'sleep 16200; timeout 60 runpodctl pod delete "$RUNPOD_POD_ID" || timeout 60 runpodctl pod stop "$RUNPOD_POD_ID"' </dev/null >/dev/null 2>&1 &
#
# DRY_RUN=1 prints the commands to the log instead of running them.
set -u

S=${1:-16200}
case "$S" in
  ''|*[!0-9]*) echo "usage: fs11_guard.sh [SECONDS]" >&2; exit 2 ;;
esac
GUARD_DRY=${DRY_RUN:-0}
LOG=${FS11_GUARD_LOG:-/dev/null}
if [ "$GUARD_DRY" != 1 ] && [ -z "${RUNPOD_POD_ID:-}" ]; then
  echo "refused: RUNPOD_POD_ID is not set, so the guard could not end the pod (DRY_RUN=1 to try it)" >&2
  exit 2
fi
GUARD_S=$S
export GUARD_S GUARD_DRY

BODY='
say() { echo "[$(date -u +%H:%M:%S)] fs11_guard: $*"; }
ctl() {
  if [ "$GUARD_DRY" = 1 ]; then say "DRY_RUN: runpodctl $*"; return 0; fi
  say "trying: runpodctl $*"
  if command -v timeout >/dev/null 2>&1; then timeout 60 runpodctl "$@"; else runpodctl "$@"; fi
}
rest_delete() {
  [ -n "${RUNPOD_API_KEY:-}" ] && command -v curl >/dev/null 2>&1 || return 1
  say "trying: REST DELETE pod $RUNPOD_POD_ID"
  echo "Authorization: Bearer $RUNPOD_API_KEY" |
    timeout 60 curl -fsS -o /dev/null -X DELETE -H @- \
      "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
}
say "armed: firing in $GUARD_S s"
sleep "$GUARD_S"
say "deadline reached"
for i in 1 2 3; do
  ctl pod delete "${RUNPOD_POD_ID:-}" && exit 0
  ctl remove pod "${RUNPOD_POD_ID:-}" && exit 0
  [ "$GUARD_DRY" != 1 ] && rest_delete && exit 0
  sleep 30
done
for i in 1 2 3; do
  ctl pod stop "${RUNPOD_POD_ID:-}" && exit 0
  sleep 30
done
say "every terminate failed"
exit 1
'

if command -v setsid >/dev/null 2>&1; then
  nohup setsid sh -c "$BODY" </dev/null >>"$LOG" 2>&1 &
else
  nohup sh -c "$BODY" </dev/null >>"$LOG" 2>&1 &
fi
echo "fs11_guard: pod ${RUNPOD_POD_ID:-(dry run)} ends in $S s; disarm with: kill $!"
