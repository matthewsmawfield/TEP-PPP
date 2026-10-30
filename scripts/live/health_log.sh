#!/bin/bash
# TEP-PPP board health logger: temp, throttle flags, volts, mem, swap,
# load.  Written for post-mortem crash correlation — the last rows
# before a freeze show whether it was undervoltage, OOM, or thermal.
CSV=/home/pi/TEP-PPP/logs/live/board_health.csv
[ -f $CSV ] || echo 'epoch_utc,temp_c,throttled_hex,core_v,mem_free_mb,swap_free_mb,load1' > $CSV
while true; do
  T=$(vcgencmd measure_temp 2>/dev/null | grep -oE '[0-9.]+')
  TH=$(vcgencmd get_throttled 2>/dev/null | grep -oE '0x[0-9a-f]+')
  V=$(vcgencmd measure_volts 2>/dev/null | grep -oE '[0-9.]+')
  MF=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  SF=$(awk '/SwapFree/{printf "%d", $2/1024}' /proc/meminfo)
  L=$(cut -d' ' -f1 /proc/loadavg)
  echo "$(date -u +%s),$T,$TH,$V,$MF,$SF,$L" >> $CSV
  sleep 60
done
