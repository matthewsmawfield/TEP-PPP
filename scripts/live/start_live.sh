#!/bin/bash
# Start the TEP-PPP live system on the Pi
cd ~/TEP-PPP
source .venv/bin/activate
set -a; source .env.local; set +a

# Kill any existing instances
pkill -f 'tep_realtime_server.py' 2>/dev/null || true
pkill -f 'gps_web.py' 2>/dev/null || true
pkill -f 'telescope_tft.py' 2>/dev/null || true
sleep 2

# Find the GPS serial port
GPS_PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -1)
if [ -z "$GPS_PORT" ]; then
    GPS_PORT=/dev/serial0
fi
echo "Using GPS port: $GPS_PORT"

# Start kriging server
OSNET_HISTORY_HOURS=14 nohup python3 -u scripts/live/tep_realtime_server.py --port 8001 > logs/live/tep_server.log 2>&1 &
KPID=$!
echo "Kriging server PID: $KPID"
sleep 15

# Start gps_web
OSNET_HISTORY_HOURS=14 nohup python3 -u rpi/gps_web.py     --port $GPS_PORT --baud 115200     --host 0.0.0.0 --web-port 8000     --kriging-url http://localhost:8001     --ref-ecef 3789582.2370,6808.5218,5113127.6141     --ppp-capture 14400 --ppp-chunk 300     > logs/live/gps_web.log 2>&1 &
WPID=$!
echo "gps_web PID: $WPID"
sleep 5

# Start telescope TFT viewer (with GPS overlay)
rm -rf /tmp/telescope_cache/*
SDL_VIDEODRIVER=dummy nohup python3 -u rpi/telescope_tft.py > /tmp/telescope.log 2>&1 &
TPID=$!
echo "Telescope viewer PID: $TPID"
sleep 3

# Check they're running
if kill -0 $KPID 2>/dev/null && kill -0 $WPID 2>/dev/null; then
    echo "All services running. Dashboard at http://$(hostname -I | awk '{print $1}'):8000"
else
    echo "ERROR: One or both services failed to start"
    tail -10 logs/live/tep_server.log 2>/dev/null
    tail -10 logs/live/gps_web.log 2>/dev/null
fi
