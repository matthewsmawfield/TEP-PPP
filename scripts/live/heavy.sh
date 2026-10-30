#!/bin/bash
# Serialize + bound heavy misc jobs (fetch, replay children, manual diags).
# macOS port: mkdir lock replaces flock (not shipped on macOS); cgroup
# memory bounds do not exist on macOS — TEP_HEAVY_HIGH/MAX are accepted
# but advisory only (documented env contract kept so callers don't change).
# Wait bounded by TEP_HEAVY_WAIT (s).
#
# Locking notes: mkdir is atomic, but unlike flock it survives the holder's
# death.  Two leak paths were observed: (a) `exec` replaces the shell so an
# EXIT trap never runs, (b) SIGKILL skips traps.  So the owner PID is stored
# in $LOCK/pid and a dead owner lets a waiter reclaim the lock.
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 1
LOCK=/tmp/tep-heavy.lockdir
WAIT=${TEP_HEAVY_WAIT:-1200}
t=0
until mkdir "$LOCK" 2>/dev/null; do
    owner=$(cat "$LOCK/pid" 2>/dev/null || echo 0)
    if ! kill -0 "$owner" 2>/dev/null; then
        # Holder gone without releasing — reclaim and retry the mkdir.
        rm -f "$LOCK/pid" 2>/dev/null
        rmdir "$LOCK" 2>/dev/null
        continue
    fi
    sleep 5; t=$((t+5))
    [ "$t" -ge "$WAIT" ] && { echo "heavy.sh: lock timeout" >&2; exit 1; }
done
echo $$ > "$LOCK/pid"
trap 'rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null' EXIT
nice -19 "$@"
