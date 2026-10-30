#!/bin/bash
# Sequential degraded-arm replay queue with a memory guard.
#
# Queue file: logs/replay/queue.txt, one job per line:
#   LABEL|CONVDIR|ARMS|ENV_ASSIGNMENTS(space-separated)
# A job is skipped when logs/replay/<label>.done exists (created only on
# rc=0).  State lives under logs/replay/ so it survives reboots — unlike
# /tmp.  Replay caches go to data/ppp_live/replay_cache (TEP_REPLAY_TMPDIR)
# for the same reason.
#
# Memory guard: a job starts only when MemAvailable >= START_MB and no
# live rnx2rtkp/convbin stage is running; the child is killed if
# MemAvailable < KILL_MB (the live pipeline outranks replays — see the
# 2026-09-28/29 brownouts in logs/argos_progress.md).
#
# Usage: nohup scripts/diag/replay_queue.sh >> logs/replay/runner.log 2>&1 &

set -u
cd "$(cd "$(dirname "$0")/../.." && pwd)"
Q=logs/replay/queue.txt
# macOS defaults: juju2020 has far more headroom than the Pi's 1GB, but
# the guard stays on so the live pipeline still outranks replays.
START_MB=${REPLAY_START_MB:-500}
KILL_MB=${REPLAY_KILL_MB:-250}
MAX_ATTEMPTS=${REPLAY_MAX_ATTEMPTS:-10}
POLL=20

# macOS port: approximate MemAvailable as free+inactive+speculative pages
# (purgeable is already inside 'free' on recent macOS).  Page size comes
# from the vm_stat header (4096 on Intel, 16384 on Apple Silicon).
memavail() {
    vm_stat | awk '
        /page size of/ { ps=$8 }
        /Pages free/ { gsub(/\./,"",$3); f=$3 }
        /Pages inactive/ { gsub(/\./,"",$3); i=$3 }
        /Pages speculative/ { gsub(/\./,"",$3); s=$3 }
        END { if (ps>0) printf "%d", (f+i+s)*ps/1048576; else print 0 }'
}
# Live-heavy detection: convbin/rnx2rtkp run as children, but the
# ~20-40 min measurement extraction and the per-arm EKF solves run
# INSIDE gps_web.  Replays launched during those stages were killed
# mid-flight every time (2026-09-30: 7 failed attempts on
# w956_harmgate).  The last "extract cache:" log line is "cold
# start"/"checkpoint" while extracting and "hit, computed" when done
# — deterministic where a point CPU sample can slip through a batch
# boundary (att 7 launched 5 s before a checkpoint write).  The CPU
# sample remains for the post-extraction EKF arm solves.
live_busy() {
    pgrep -f "rnx2rtkp|convbin" >/dev/null 2>&1 && return 0
    # Pending cold start: the cfg hash covers precise product CONTENT, so
    # any *.sp3/*.clk newer than the extract cache guarantees a cold
    # start on the next solve.  The log check below only fires AFTER
    # gps_web commits to it — a replay launched in the gap rode the
    # 16 s race at 06:59 UTC (w32266_base att 2, killed 22 min in by
    # the thrash pairing).  Missing pkl also means a cold start is due.
    local _ec="data/ppp_live/extract_cache.pkl"
    if [ ! -f "$_ec" ]; then
        return 0
    fi
    [ -n "$(find data/ppp_live/precise -maxdepth 1 \
        \( -name '*.sp3' -o -name '*.clk' \) -newer "$_ec" \
        -print -quit 2>/dev/null)" ] && return 0
    case "$(tail -c 2000000 logs/live/gps_web.log 2>/dev/null \
            | grep -a 'extract cache:' | tail -1)" in
        *"cold start"*|*checkpoint*) return 0;;
    esac
    # Arm-solve gate: a "PPP EKF INIT" with no later "pos_rmse=" in the
    # log tail means a window pipeline (calib -> persat -> arm solves ->
    # fullsky) is in flight.  The 2 s CPU sample slips through the
    # EKF-silent calibration/persat gap between INITs (2026-09-30 run
    # 56: three w956_harmgate attempts launched mid-solve at 15:21/15:33
    # and were KILLed at ~50 MB — the same thrash regime that caused the
    # watchdog reset).  INIT-vs-pos_rmse line position is deterministic.
    local _tail2m _il _dl
    _tail2m=$(tail -c 2000000 logs/live/gps_web.log 2>/dev/null | tr -d '\0')
    _il=$(printf '%s\n' "$_tail2m" | grep -an 'EKF INIT' | tail -1 | cut -d: -f1)
    _dl=$(printf '%s\n' "$_tail2m" | grep -an 'pos_rmse=' | tail -1 | cut -d: -f1)
    [ -n "$_il" ] && { [ -z "$_dl" ] || [ "$_il" -gt "$_dl" ]; } && return 0
    # Post-restart grace: a fresh gps_web runs the full window pipeline
    # (convbin -> extract -> arm solves) and the extract-cache line may
    # not exist yet for the first ~90 s — that gap let a replay launch
    # 30 s into the 2026-09-30 09:12 restart and collide on memory.
    # Treat a young gps_web as busy.
    local pid start_s
    pid=$(pgrep -f "rpi/gps_web.py" | head -1)
    if [ -n "$pid" ]; then
        # macOS ps has no 'etimes' — derive age from lstart via BSD date -j.
        start_s=$(date -j -f "%a %b %d %T %Y" "$(ps -o lstart= -p "$pid" 2>/dev/null)" +%s 2>/dev/null || echo 0)
        [ "$start_s" -gt 0 ] && [ $(( $(date +%s) - start_s )) -lt "${REPLAY_RESTART_GRACE_S:-1800}" ] \
            && return 0
    fi
    # macOS: no /proc — sample cumulative CPU time via ps -o time= twice.
    # Threshold default 390 cs (~1.95 cores in 2 s): the live per-second
    # EKF now burns ~190 cs steady-state (LUPT/HUPT rows, 40+ sats) so
    # the old 150 cs bar tripped permanently and starved the queue for
    # 3.7 h on 2026-10-15.  390 still catches real bursts (conv stage +
    # python together) while gates 1-4 cover the heavy stages
    # deterministically.
    local pid a b
    pid=$(pgrep -f "rpi/gps_web.py" | head -1)
    [ -n "$pid" ] || return 1
    cpu_cs() {  # ps time [[dd-]hh:]mm:ss.hh -> centiseconds
        ps -o time= -p "$1" 2>/dev/null | awk -F'[-:.]' '{
            n=split($0,p,/[-:.]/); s=0
            if (n==3) s=p[1]*60*100+p[2]*100+p[3]
            else if (n==4) s=p[1]*3600*100+p[2]*60*100+p[3]*100+p[4]
            else if (n==5) s=p[1]*86400*100+p[2]*3600*100+p[3]*60*100+p[4]*100+p[5]
            print s+0 }'
    }
    a=$(cpu_cs "$pid") || return 1
    sleep 2
    b=$(cpu_cs "$pid") || return 1
    [ $((b - a)) -ge "${REPLAY_CPU_BUSY_CS:-390}" ]
}

while true; do
    # Find next undone job
    job=""
    while IFS= read -r line; do
        case "$line" in ''|\#*) continue;; esac
        lbl=${line%%|*}
        [ -f "logs/replay/${lbl}.done" ] && continue
        job=$line; break
    done < "$Q"
    [ -z "$job" ] && { echo "$(date -u +%FT%T) queue empty — exiting"; exit 0; }

    lbl=${job%%|*}; rest=${job#*|}
    conv=${rest%%|*}; rest=${rest#*|}
    arms=${rest%%|*}; envs=${rest#*|}
    att=$(cat "logs/replay/${lbl}.att" 2>/dev/null || echo 0)
    if [ "$att" -ge "$MAX_ATTEMPTS" ]; then
        echo "$(date -u +%FT%T) $lbl: $att attempts — giving up"
        touch "logs/replay/${lbl}.done"
        continue
    fi

    # Wait for memory headroom AND no live heavy stage
    while :; do
        m=$(memavail)
        if [ "$m" -ge "$START_MB" ] && ! live_busy; then break; fi
        sleep $POLL
    done
    echo "$(date -u +%FT%T) $lbl: starting (att=$((att+1)), mem=${m}MB)"

    set -a; [ -f .env.local ] && . ./.env.local; set +a
    # setsid makes the job its own process-group leader so the memory
    # guard can kill the WHOLE tree (eval subshell + python + any
    # rnx2rtkp children).  2026-09-29: kill -9 $pid through eval left an
    # orphaned python that kept running unmonitored for 35min.
    # Cap BLAS/OpenMP thread pools: the arms' numpy ops are per-epoch
    # and small, so a 4-thread pool buys nothing, but each thread's
    # buffers inflate RSS (~50-100MB observed) and steal cores from the
    # live pipeline.  Serial threads + nice -19 = fewer mem-kills and
    # less contention with a running solve.
    # macOS has no setsid; bash runs async children in their own pgid
    # under a non-interactive shell only with job control, so we rely on
    # the pkill -f fallback below for tree-kill instead of kill -- -$pid.
    env $envs TEP_REPLAY_TMPDIR=data/ppp_live/replay_cache \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        TEP_HEAVY_HIGH=300M TEP_HEAVY_MAX=400M TEP_HEAVY_WAIT=3600 scripts/live/heavy.sh .venv/bin/python3 scripts/attic/diag_arm_replay.py "$conv" "$arms" \
        > "logs/replay/${lbl}.log" 2>&1 &
    pid=$!

    killed=0
    while kill -0 $pid 2>/dev/null; do
        m=$(memavail)
        if [ "$m" -lt "$KILL_MB" ]; then
            echo "$(date -u +%FT%T) $lbl: KILL mem=${m}MB"
            kill -9 $pid 2>/dev/null
            # reap stragglers that ignored the group signal
            sleep 2
            pkill -9 -f "diag_arm_replay.py $conv" 2>/dev/null
            killed=1; break
        fi
        sleep 10
    done
    wait $pid; rc=$?
    if [ $rc -eq 0 ] && [ $killed -eq 0 ]; then
        echo "$(date -u +%FT%T) $lbl: DONE"
        touch "logs/replay/${lbl}.done"
    elif grep -q "obs file is gone\|cannot finish extraction\|No such file\|SPP failed\|no obs files\|cannot replay" \
            "logs/replay/${lbl}.log" 2>/dev/null; then
        # Terminal failure: the specimen's raw data was pruned or its SPP
        # is deterministically unsolvable, so every retry fails
        # identically.  Tombstone now instead of burning all MAX_ATTEMPTS.
        echo "$(date -u +%FT%T) $lbl: obs data pruned/unusable — giving up (rc=$rc)"
        touch "logs/replay/${lbl}.done"
    else
        echo $((att+1)) > "logs/replay/${lbl}.att"
        echo "$(date -u +%FT%T) $lbl: failed rc=$rc killed=$killed (att=$((att+1)))"
        sleep 60
    fi
done
