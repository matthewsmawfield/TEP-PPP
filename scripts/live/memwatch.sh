#!/bin/bash
# Periodic memory + power snapshot: separates true leaks from peak-demand
# crunches AND logs voltage sag history (0x50000 = undervoltage+throttle
# have occurred since boot). One line per 5min via cron.
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
AVAIL=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
SWAPU=$(awk "/SwapFree/{print int(\$2/1024)}" /proc/meminfo)
GW=$(awk "/VmRSS/{print int(\$2/1024)}" /proc/$(pgrep -f "rpi/gps_web.py" | head -1)/status 2>/dev/null)
KR=$(awk "/VmRSS/{print int(\$2/1024)}" /proc/$(pgrep -f "tep_realtime_server" | head -1)/status 2>/dev/null)
THR=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
VOL=$(vcgencmd measure_volts core 2>/dev/null | cut -d= -f2)
echo "$TS avail=${AVAIL}M swapfree=${SWAPU}M gps_web=${GW:-0}M kriging=${KR:-0}M thr=$THR vcore=$VOL"
