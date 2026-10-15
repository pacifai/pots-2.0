#!/usr/bin/env bash
# FS11's H100 session as one unattended job, which ends its own pod.
#
#   setsid nohup bash verification/runs/fs11_job.sh UPLOAD_DIR OUT_DIR \
#       </dev/null > /workspace/fs11_job.out 2>&1 &
#
# Run from the repo (the upload's repo/, with its .venv). The pod's start command arms a boot
# guard (verification/README.md, "Workflow: the first H100 session"), which ends the pod if
# setup stalls or this job never starts; fs11_guard.sh re-arms it by hand. FS11_SESSION picks
# the session's plan: fs11 (the default), the first session's, or recheck, the re-check's
# (verification/runs/fs11_session.py); anything else is refused. In order:
#
# 1. Preflight. Without DRY_RUN the job refuses to start unless it can terminate the pod:
#    RUNPOD_POD_ID set, and runpodctl or RUNPOD_API_KEY there (exit 2). A refusal it can act on
#    (bad deadlines, less than FS11_MIN_FREE_GB free under OUT_DIR) terminates the pod.
# 2. A deadline timer starts: FS11_DEADLINE_S (default 12600, 3.5 h; 7200, 2 h, for the
#    re-check) after the start it terminates the pod, whatever the job is doing.
# 3. The session (verification.runs.fs11_session run) prints the environment, refuses a
#    machine without an H100 (FS11_ALLOW_ANY_GPU=1 overrides), and runs its parts, each with
#    its own log under OUT_DIR. Each part's exit code is recorded and a failed part doesn't stop
#    the others. A running part is killed at the work deadline, which leaves FS11_GRACE_S plus
#    FS11_PACK_RESERVE_S before the timer fires. A session still running FS11_HANG_MARGIN_S
#    (default 120) after the work deadline is killed, with its running part, so packing starts.
# 4. Whatever the session's outcome, OUT_DIR is packed into OUT_DIR.tar, its SHA-256 printed and
#    written to OUT_DIR.tar.sha256, and OUT_DIR.READY written: the tarball can be fetched. If
#    packing fails, OUT_DIR.PACK_FAILED is written instead.
# 5. A grace window of FS11_GRACE_S (default 1200, 20 min) for the copy. Touching
#    OUT_DIR.FETCHED ends it early.
# 6. The pod is terminated (removed, not stopped). SIGTERM or SIGINT to the job terminates it
#    too. SIGHUP can't reach it: setsid detaches it from the terminal, and nohup ignores HUP
#    (an ignored signal can't be trapped).
#
# Logging. Everything goes to OUT_DIR/job.log, appended to directly, with no logger process
# between: a dead reader can't take the job down. SIGPIPE is ignored throughout. The job's own
# lines are also copied to its stdout (fs11_job.out), best effort.
#
# Terminating. Every runpodctl and curl call runs under a time limit (FS11_CALL_TIMEOUT_S,
# default 60). OUT_DIR.TERMINATED is written only after a delete returned success and a
# follow-up read confirmed the pod gone (REST GET answers 404, or runpodctl pod get says not
# found), or no read was possible. After FS11_DELETE_ROUNDS (default 3) rounds of failed deletes,
# each round also tries runpodctl pod stop and then the REST stop. Rounds repeat every
# FS11_RETRY_S (default 30): the job's own call until the deadline, then the timer's without
# end. A lock directory, OUT_DIR.terminate.lock, keeps the job and the timer from terminating
# at the same time, and the timer is killed only after a terminate succeeded. The key reaches
# curl on stdin (--config -), never on a command line.
#
# DRY_RUN=1 prints the terminate commands instead of running them, and kills the job when the
# timer fires, as the pod's removal would. FS11_SELFTEST=1 runs a stub plan (no GPU check, no
# real part) under a 360 s deadline with a 120 s grace window, then terminates for real
# through the normal end-of-run path; the log names the method that worked. It's for a cheap
# CPU pod, to check that the pod-scoped key may delete its own pod.
#
# Every run writes OUT_DIR/terminate_tools.txt before the session starts, and copies it to the
# log: where runpodctl is and its version, whether RUNPOD_API_KEY is set (never its value),
# and whether curl, timeout and setsid are there.
#
# Runpod's commands, from its public docs (checked 2026-10-14):
# - https://docs.runpod.io/pods/manage-pods: "runpodctl pod delete $RUNPOD_POD_ID" terminates
#   a pod, which "permanently deletes all data not stored in a network volume";
#   "runpodctl pod stop" only stops it, and a stopped pod still bills for its disk.
# - https://docs.runpod.io/runpodctl/reference/runpodctl-pod: "runpodctl pod get <pod-id>",
#   "runpodctl pod stop <pod-id>", "runpodctl pod delete <pod-id>". What get prints for a
#   missing pod isn't documented; a non-zero exit saying "not found" is taken as gone.
# - https://docs.runpod.io/pods/references/environment-variables: RUNPOD_POD_ID is the
#   "Unique Pod identifier", RUNPOD_API_KEY the "Pod-scoped API key".
# - https://docs.runpod.io/runpodctl/overview: "Every Pod you deploy comes preinstalled with
#   runpodctl and a Pod-scoped API key" (runpodctl 2.x).
# - Fallbacks: runpodctl 1.x spells it "runpodctl remove pod <podId>"
#   (github.com/runpod/runpodctl, docs/runpodctl_remove_pod.md). The REST API, bearer key:
#   "DELETE https://rest.runpod.io/v1/pods/{podId}"
#   (https://docs.runpod.io/api-reference/pods/DELETE/pods/podId), "GET .../pods/{podId}" with
#   404 "Pod not found" (.../GET/pods/podId), "POST .../pods/{podId}/stop" (.../POST/pods/
#   podId/stop).
set -u
trap '' PIPE

UPLOAD=${1:?usage: fs11_job.sh UPLOAD_DIR OUT_DIR}
OUT=${2:?usage: fs11_job.sh UPLOAD_DIR OUT_DIR}
SELFTEST=${FS11_SELFTEST:-0}
SESSION=${FS11_SESSION:-fs11}
DEFAULT_DEADLINE_S=12600
[ "$SESSION" = recheck ] && DEFAULT_DEADLINE_S=7200
if [ "$SELFTEST" = 1 ]; then
  DEADLINE_S=${FS11_DEADLINE_S:-360}; GRACE_S=${FS11_GRACE_S:-120}
  PACK_RESERVE_S=${FS11_PACK_RESERVE_S:-60}; MIN_FREE_GB=${FS11_MIN_FREE_GB:-1}
  export FS11_ALLOW_ANY_GPU=1
else
  DEADLINE_S=${FS11_DEADLINE_S:-$DEFAULT_DEADLINE_S}; GRACE_S=${FS11_GRACE_S:-1200}
  PACK_RESERVE_S=${FS11_PACK_RESERVE_S:-600}; MIN_FREE_GB=${FS11_MIN_FREE_GB:-15}
fi
HANG_MARGIN_S=${FS11_HANG_MARGIN_S:-120}
STOP_GRACE_S=${FS11_STOP_GRACE_S:-45}  # the session's own part kill takes up to 40 s
CALL_TIMEOUT_S=${FS11_CALL_TIMEOUT_S:-60}
RETRY_S=${FS11_RETRY_S:-30}
DELETE_ROUNDS=${FS11_DELETE_ROUNDS:-3}
CONFIRM_S=${FS11_CONFIRM_S:-60}
PY=${FS11_PYTHON:-.venv/bin/python}
if [ "$SELFTEST" = 1 ] && [ -z "${FS11_PYTHON:-}" ] && [ ! -x "$PY" ]; then
  PY=$(command -v python3 || command -v python || echo python3)  # a bare pod: no venv
fi
DRY=${DRY_RUN:-0}
POD=${RUNPOD_POD_ID:-}

case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
case "$UPLOAD" in /*) ;; *) UPLOAD="$PWD/$UPLOAD" ;; esac
OUT=${OUT%/}
TAR="$OUT.tar"
READY="$OUT.READY"
PACK_FAILED="$OUT.PACK_FAILED"
FETCHED="$OUT.FETCHED"
FLAG="$OUT.TERMINATED"
LOCK="$OUT.terminate.lock"
MIRROR=0  # 1 once job.log takes stdout: say() then copies its line to fd 3 too

exec 3>&1
# say writes through an external echo. A builtin's write that fails (a dead reader on the
# other end) stays in bash's output buffer and turns up in a later command substitution.
ECHO=/bin/echo
[ -x "$ECHO" ] || ECHO=/usr/bin/echo
say() {
  local line
  line="[$(date -u +%H:%M:%S)] $*"
  "$ECHO" "$line" 2>/dev/null
  if [ "$MIRROR" = 1 ]; then "$ECHO" "$line" >&3 2>/dev/null; fi
  return 0
}

utc() { date -u -d "@$1" +%H:%M:%SZ 2>/dev/null || date -u -r "$1" +%H:%M:%SZ; }

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# A command under the call time limit: coreutils' timeout on the pod, perl's alarm elsewhere.
tmo() {
  if command -v timeout >/dev/null 2>&1; then timeout -k 5 "$CALL_TIMEOUT_S" "$@"
  else perl -e '$t = shift; alarm $t; exec @ARGV or exit 127' "$CALL_TIMEOUT_S" "$@"; fi
}

have_ctl() { command -v runpodctl >/dev/null 2>&1; }
have_rest() { [ -n "${RUNPOD_API_KEY:-}" ] && command -v curl >/dev/null 2>&1; }

# rest METHOD SUFFIX: one REST call on this pod; prints the HTTP status (000 when none came).
rest() {
  printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" |
    tmo curl -sS -o /dev/null -w '%{http_code}' -X "$1" --config - \
      "https://rest.runpod.io/v1/pods/$POD$2" 2>/dev/null
}

# 0 when a read shows the pod gone, or no read is possible; 1 when it is still there after
# CONFIRM_S. A deleted pod takes this container with it, so the wait mostly never ends here.
confirm_gone() {
  local t_end code out rc seen
  t_end=$(($(date +%s) + CONFIRM_S))
  while :; do
    seen=unknown
    if have_rest; then
      code=$(rest GET "")
      case "$code" in
        404) say "confirmed: GET pod $POD answers 404"; return 0 ;;
        200) seen=there ;;
      esac
    fi
    if [ "$seen" = unknown ] && have_ctl; then
      out=$(tmo runpodctl pod get "$POD" 2>&1); rc=$?
      if [ $rc -ne 0 ] && printf '%s' "$out" | grep -qi 'not found\|404\|does not exist'; then
        say "confirmed: runpodctl pod get $POD says not found"; return 0
      fi
      [ $rc -eq 0 ] && seen=there
    fi
    if [ "$seen" = unknown ]; then
      say "no read can confirm the pod's end; taking the call's success"; return 0
    fi
    [ "$(date +%s)" -ge "$t_end" ] && { say "pod $POD still listed ${CONFIRM_S} s later"; return 1; }
    sleep 5
  done
}

try_delete() {
  local code
  if have_ctl; then
    say "trying: runpodctl pod delete $POD"
    tmo runpodctl pod delete "$POD" && confirm_gone && { METHOD="runpodctl pod delete"; return 0; }
    say "trying: runpodctl remove pod $POD"
    tmo runpodctl remove pod "$POD" && confirm_gone && { METHOD="runpodctl remove pod"; return 0; }
  fi
  if have_rest; then
    say "trying: REST DELETE pod $POD"
    code=$(rest DELETE "")
    case "$code" in
      200|202|204) confirm_gone && { METHOD="REST DELETE"; return 0; } ;;
      *) say "REST DELETE answered HTTP $code" ;;
    esac
  fi
  return 1
}

try_stop() {
  local code
  if have_ctl; then
    say "trying: runpodctl pod stop $POD"
    tmo runpodctl pod stop "$POD" && { METHOD="runpodctl pod stop (stopped, not removed)"; return 0; }
  fi
  if have_rest; then
    say "trying: REST POST pod $POD/stop"
    code=$(rest POST /stop)
    case "$code" in
      200|202|204) METHOD="REST stop (stopped, not removed)"; return 0 ;;
      *) say "REST stop answered HTTP $code" ;;
    esac
  fi
  return 1
}

# terminate REASON [UNTIL]: end the pod, retrying until the epoch UNTIL (0: without end).
# Returns 0 once OUT_DIR.TERMINATED is written, by this caller or the other.
terminate() {
  local reason=$1 until=${2:-0} me holder round=0 waited=0
  me=${BASHPID:-$$}  # the timer's subshell, or the job
  TERMINATING=1
  while ! mkdir "$LOCK" 2>/dev/null; do
    [ -e "$FLAG" ] && { TERMINATING=0; return 0; }
    holder=$(cat "$LOCK/pid" 2>/dev/null)
    [ "$waited" = 0 ] && say "terminate ($reason) waits for the other terminate (pid ${holder:-?})"
    waited=1
    if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then rm -rf "$LOCK"; continue; fi
    if [ "$until" -gt 0 ] && [ "$(date +%s)" -ge "$until" ]; then TERMINATING=0; return 1; fi
    sleep 1
  done
  echo "$me" > "$LOCK/pid"
  if [ -e "$FLAG" ]; then rm -rf "$LOCK"; TERMINATING=0; return 0; fi
  if [ "$DRY" = 1 ]; then
    say "DRY_RUN: terminate ($reason): runpodctl pod delete ${POD:-\$RUNPOD_POD_ID}"
    say "DRY_RUN:   fallbacks: runpodctl remove pod ${POD:-\$RUNPOD_POD_ID}; curl -X DELETE" \
        "https://rest.runpod.io/v1/pods/${POD:-\$RUNPOD_POD_ID} (key on stdin); then" \
        "runpodctl pod stop, POST .../stop"
    METHOD="dry run"
  else
    say "terminating pod $POD ($reason)"
    while :; do
      round=$((round + 1))
      try_delete && break
      if [ "$round" -ge "$DELETE_ROUNDS" ]; then try_stop && break; fi
      if [ "$until" -gt 0 ] && [ $(($(date +%s) + RETRY_S)) -ge "$until" ]; then
        say "TERMINATE FAILED after $round rounds; the deadline timer goes on trying"
        rm -rf "$LOCK"; TERMINATING=0; return 1
      fi
      say "terminate round $round failed; retrying in $RETRY_S s"
      sleep "$RETRY_S"
    done
  fi
  echo "$reason; method: $METHOD; $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$FLAG"
  say "terminated by: $METHOD"
  rm -rf "$LOCK"
  TERMINATING=0
  return 0
}

# A refusal: terminate when the pod can be ended (or in a dry run), then exit 2.
refuse() {
  say "refused: $*"
  if [ "$DRY" = 1 ] || { [ -n "$POD" ] && { have_ctl || have_rest; }; }; then
    terminate "refused: $*" 0
  fi
  exit 2
}

# ---- before anything ---------------------------------------------------------------------
START=$(date +%s)
HARD=$((START + DEADLINE_S))
WORK_DEADLINE=$((HARD - GRACE_S - PACK_RESERVE_S))
mkdir -p "$(dirname "$OUT")"
rm -rf "$LOCK"
rm -f "$FLAG"
if [ "$DRY" != 1 ]; then
  if [ -z "$POD" ]; then
    say "refused: RUNPOD_POD_ID is not set, so the job could not end its pod (DRY_RUN=1 to try it)"
    exit 2
  fi
  if ! have_ctl && [ -z "${RUNPOD_API_KEY:-}" ]; then
    say "refused: neither runpodctl nor RUNPOD_API_KEY is there to terminate pod $POD"
    exit 2
  fi
fi
if [ $((GRACE_S + PACK_RESERVE_S)) -ge "$DEADLINE_S" ]; then
  refuse "FS11_GRACE_S + FS11_PACK_RESERVE_S must be below FS11_DEADLINE_S"
fi
case "$SESSION" in
  fs11|recheck) ;;
  *) refuse "FS11_SESSION is '$SESSION'; it must be fs11 or recheck" ;;
esac
mkdir -p "$OUT" || refuse "cannot create $OUT"
rm -f "$READY" "$PACK_FAILED" "$FETCHED"
FREE_KB=$(df -Pk "$OUT" 2>/dev/null | awk 'NR == 2 {print $4}')
if [ -n "$FREE_KB" ] && [ "$FREE_KB" -lt $((MIN_FREE_GB * 1048576)) ]; then
  refuse "$((FREE_KB / 1048576)) GiB free under $OUT, FS11_MIN_FREE_GB is $MIN_FREE_GB"
fi
if exec >>"$OUT/job.log" 2>&1; then MIRROR=1
else say "cannot append to $OUT/job.log; logging to stdout only"; fi

MAIN=$$
say "FS11 job: session $SESSION, upload $UPLOAD, out $OUT, pod ${POD:-none}, DRY_RUN=$DRY, SELFTEST=$SELFTEST"
say "deadline in ${DEADLINE_S} s ($(utc "$HARD")); parts stop by $((WORK_DEADLINE - START)) s; grace ${GRACE_S} s; ${FREE_KB:-?} KiB free"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_TOKEN_PATH
if command -v nvidia-smi >/dev/null 2>&1; then tmo nvidia-smi 3>&-; fi
where() { command -v "$1" 2>/dev/null || echo none; }
{
  echo "runpodctl=$(where runpodctl)"
  if have_ctl; then
    echo "runpodctl_version=$(tmo runpodctl version 2>&1 3>&- | head -n 3 | tr '\n' ' ')"
  fi
  echo "runpod_api_key=$([ -n "${RUNPOD_API_KEY:-}" ] && echo set || echo unset)"
  echo "runpod_pod_id=${POD:-unset}"
  echo "curl=$(where curl)"
  echo "timeout=$(where timeout)"
  echo "setsid=$(where setsid)"
} > "$OUT/terminate_tools.txt" 2>/dev/null
say "terminate tools: $(tr '\n' ' ' < "$OUT/terminate_tools.txt" 2>/dev/null)"

# ---- the deadline timer ------------------------------------------------------------------
(
  trap '' PIPE
  trap 'kill ${SLEEP:-} 2>/dev/null; exit 0' TERM
  sleep $((HARD - $(date +%s))) & SLEEP=$!
  wait $SLEEP
  # Terminate first, log after: nothing may stand between the deadline and the pod's end.
  terminate "deadline" 0
  say "deadline reached after ${DEADLINE_S} s"
  # The pod is gone by now. In a dry run (or if this container outlived its pod), end the
  # job as the removal would; its trap stops the session.
  kill -TERM "$MAIN" 2>/dev/null
) &
TIMER=$!

# Stop the session and its running part: TERM, then KILL after STOP_GRACE_S.
stop_session() {
  local pg t_end
  [ -n "${ORCH:-}" ] || return 0
  kill -TERM "$ORCH" 2>/dev/null
  t_end=$(($(date +%s) + STOP_GRACE_S))
  while kill -0 "$ORCH" 2>/dev/null && [ "$(date +%s)" -lt "$t_end" ]; do sleep 1; done
  pg=$(cat "$OUT/current_part" 2>/dev/null)
  kill -KILL "$ORCH" 2>/dev/null
  if [ -n "$pg" ]; then
    kill -TERM -- "-$pg" 2>/dev/null && sleep 2
    kill -KILL -- "-$pg" 2>/dev/null
  fi
  return 0
}

on_signal() {
  # A signal that lands while this shell is terminating leaves that call to finish.
  [ "${TERMINATING:-0}" = 1 ] && return
  say "job stopped by a signal"
  stop_session
  terminate "signal" 0
  kill "$TIMER" 2>/dev/null
  exit "${RC:-1}"
}
trap on_signal TERM INT

# ---- the session -------------------------------------------------------------------------
say "session: $PY -m verification.runs.fs11_session run --session $SESSION"
"$PY" -m verification.runs.fs11_session run --upload "$UPLOAD" --out "$OUT" \
  --deadline "$WORK_DEADLINE" --session "$SESSION" ${FS11_PLAN:+--plan "$FS11_PLAN"} \
  $([ "$SELFTEST" = 1 ] && echo --selftest) 3>&- &
ORCH=$!
KILL_AT=$((WORK_DEADLINE + HANG_MARGIN_S))
while kill -0 "$ORCH" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$KILL_AT" ]; then
    say "session still running ${HANG_MARGIN_S} s after the work deadline: killing it and its part"
    stop_session
    break
  fi
  sleep 2 & wait $!
done
wait "$ORCH"
RC=$?
ORCH=
say "session exit $RC"
echo "$RC" > "$OUT/session_exit_code"

# ---- pack, publish, grace, terminate -----------------------------------------------------
say "packing $OUT"
SUM=
if tar -cf "$TAR.tmp" -C "$(dirname "$OUT")" "$(basename "$OUT")" && mv "$TAR.tmp" "$TAR"; then
  SUM=$(sha256 "$TAR")
fi
NOW=$(date +%s)
END=$((NOW + GRACE_S))
[ "$END" -gt "$HARD" ] && END=$HARD
if [ -n "$SUM" ]; then
  echo "$SUM  $(basename "$TAR")" > "$TAR.sha256"
  {
    echo "tarball=$TAR"
    echo "sha256=$SUM"
    echo "bytes=$(wc -c < "$TAR" | tr -d ' ')"
    echo "session_exit=$RC"
    echo "terminate_at_epoch=$END"
  } > "$READY"
  say "READY: $TAR sha256 $SUM; terminate in $((END - NOW)) s (touch $FETCHED to end sooner)"
else
  rm -f "$TAR.tmp"
  { echo "session_exit=$RC"; echo "terminate_at_epoch=$END"; } > "$PACK_FAILED"
  say "PACK FAILED: no tarball; $OUT stays for $((END - NOW)) s (touch $FETCHED to end sooner)"
fi
while [ ! -e "$FETCHED" ]; do
  LEFT=$((END - $(date +%s)))
  [ "$LEFT" -le 0 ] && break
  sleep $((LEFT < 5 ? LEFT : 5))
done
[ -e "$FETCHED" ] && say "fetch confirmed"
if terminate "end of job (session exit $RC)" "$HARD"; then
  kill "$TIMER" 2>/dev/null
  pkill -P "$TIMER" 2>/dev/null
  exit "$RC"
fi
# The job's own attempts ran out at the deadline; the timer goes on until one succeeds.
wait "$TIMER"
exit "$RC"
