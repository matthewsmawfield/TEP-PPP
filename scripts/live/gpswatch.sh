#!/bin/bash
# gpswatch.sh — tep-gps-web liveness watchdog (macOS launchd, every 5 min).
#
# Same failure mode as the Pi cron version: the gps_web process can stay
# alive while producing no data (dead/absent serial bridge). KeepAlive
# cannot see that. Signal: /api/fix utc_time is NMEA-derived — it freezes
# the moment receiver bytes stop. Stale/missing > STALE_S (with
# process-age grace) -> SIGTERM; launchd KeepAlive + the wrapper's serial
# probe re-seat it. While the wrapper is still probing for a port (no
# gps_web.py process yet), launchd owns the retry loop — do nothing.

LOG="$HOME/TEP-PPP/logs/live/gpswatch.log"
FAIL=/tmp/gpswatch_apifail
STALE_S=900     # 15 min of frozen/missing NMEA time
GRACE_S=600     # don't judge a process younger than this
APIFAIL_N=3     # consecutive dead-API checks before killing
PYBIN="$HOME/TEP-PPP/.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN=python3   # /usr/bin/python3 is a stub when Xcode CLT is absent

pid=$(pgrep -f "rpi/gps_web.py --port" | head -1)
if [ -z "$pid" ]; then
    echo 0 > "$FAIL"
    exit 0
fi

now=$(date +%s)
# macOS ps has no 'etimes'; derive age from lstart via BSD date -j.
start_s=$(date -j -f "%a %b %d %T %Y" "$(ps -o lstart= -p "$pid" 2>/dev/null)" +%s 2>/dev/null || echo 0)
if [ "$start_s" -gt 0 ] && [ $(( now - start_s )) -lt "$GRACE_S" ]; then
    echo 0 > "$FAIL"
    exit 0
fi

# A busy solve keeps the process so CPU-saturated that the HTTP API
# can't answer inside the curl timeout, and the /api/fix NMEA-derived
# utc fields come back empty/stale while the capture thread starves —
# both conditions fire during HEALTHY multi-hour window solves.  Kill
# history 2026-10-21: nine mid-solve SIGTERMs in one night, every
# result lost (api_dead x3 during extract/arm peaks, nmea_stale utc=0
# during capture starvation).  Proof of life: the solve writes the
# log every few seconds and the capture worker appends a chunk_*.bin
# every ~300 s.  If either is fresh, the service is working — never
# a kill candidate, whatever /api/fix says.
log_fresh() {
    local lf="$HOME/TEP-PPP/logs/live/gps_web.log" lm
    [ -f "$lf" ] || return 1
    lm=$(stat -f %m "$lf" 2>/dev/null || echo 0)
    [ $(( now - lm )) -lt 300 ]
}

chunks_fresh() {
    local cm
    cm=$(stat -f %m "$HOME"/TEP-PPP/data/ppp_live/chunk_*.bin 2>/dev/null | sort -rn | head -1)
    [ -n "$cm" ] && [ $(( now - cm )) -lt 900 ]
}

alive() { log_fresh || chunks_fresh; }

kill_gw() {
    local why="$1"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) KILL gps_web pid=${pid:-none} why=$why" >> "$LOG"
    [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null
    echo 0 > "$FAIL"
}

json=$(curl -sf -m 8 http://localhost:8000/api/fix 2>/dev/null)
if [ -z "$json" ]; then
    n=$(( $(cat "$FAIL" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$FAIL"
    if [ "$n" -ge "$APIFAIL_N" ]; then
        if alive; then
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SKIP api_dead x$n — log/chunks fresh, solve in progress" >> "$LOG"
        else
            kill_gw "api_dead x$n"
        fi
    fi
    exit 0
fi
echo 0 > "$FAIL"

# Parse NMEA-derived utc_date/utc_time -> epoch (portable, no GNU date).
ts_s=$(printf '%s' "$json" | "$PYBIN" -c '
import json, sys, datetime, calendar
try:
    d = json.load(sys.stdin)
    ts = d.get("utc_date", "") + " " + d.get("utc_time", "")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(ts.strip(), fmt)
            print(calendar.timegm(dt.timetuple())); break
        except ValueError:
            continue
    else:
        print(0)
except Exception:
    print(0)' 2>/dev/null || echo 0)
stale=$(( now - ts_s ))
if [ "$ts_s" -le 0 ] || [ "$stale" -gt "$STALE_S" ]; then
    if chunks_fresh; then
        # Serial bytes are flowing (chunks arriving) — the stale/empty
        # utc fields are a parse/API artifact of a starved NMEA demux,
        # not a dead serial bridge.  utc=0 kills on 2026-10-21 were all
        # this false positive.
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SKIP nmea_stale=${stale}s utc=${ts_s} — chunks fresh, serial alive" >> "$LOG"
    else
        kill_gw "nmea_stale=${stale}s utc=${ts_s}"
    fi
fi
