#!/usr/bin/env bash
# =============================================================================
#  TEP-PPP Raspberry Pi Setup Script
#  Target: Raspberry Pi 3 Model B (1.2 GHz quad-core, 1 GB RAM)
#  GPS:    u-blox LEA-6H (initial) / Waveshare LC29H(BS) HAT (later)
#  Display: Adafruit 2.8" PiTFT (ILI9341, 240x320, SPI)
# =============================================================================
#  Run this ON the Raspberry Pi:
#    sudo bash setup_pi.sh
#
#  It configures:
#    1. Serial port (UART) for GPS module
#    2. Adafruit 2.8" PiTFT display (ILI9341 over SPI)
#    3. RTKLIB (convbin, rnx2rtkp) from source
#    4. Python environment + TEP-PPP dependencies
#    5. systemd service for the TEP Pi daemon
# =============================================================================
set -euo pipefail

# --- Colours -----------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$1"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$1"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$1"; }
err()   { printf "${RED}[ERR]${NC}   %s\n" "$1"; }

if [[ $EUID -ne 0 ]]; then
  err "Run as root: sudo bash setup_pi.sh"
  exit 1
fi

PI_USER="${SUDO_USER:-pi}"
PI_HOME="/home/$PI_USER"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

info "TEP-PPP Raspberry Pi Setup"
info "User:   $PI_USER"
info "Home:   $PI_HOME"
info "Repo:   $REPO_DIR"
echo ""

# -----------------------------------------------------------------------------
# 1. System update
# -----------------------------------------------------------------------------
info "Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq
ok "System updated"

# -----------------------------------------------------------------------------
# 2. Serial port (UART) for GPS
# -----------------------------------------------------------------------------
info "Configuring serial port for GPS..."

# Disable serial console (bluetooth on Pi 3 uses mini UART, GPS uses PL011)
raspi-config nonint do_serial 2  # disable shell on serial, enable hardware port

# On Pi 3B, bluetooth uses the mini UART. To give GPS the full PL011 UART:
# Add dtoverlay to swap bluetooth to mini-uart
CONFIG_FILE="/boot/config.txt"
if ! grep -q "dtoverlay=pi3-miniuart-bt" "$CONFIG_FILE"; then
  echo "" >> "$CONFIG_FILE"
  echo "# Pi 3B: route PL011 UART to GPIO pins for GPS" >> "$CONFIG_FILE"
  echo "dtoverlay=pi3-miniuart-bt" >> "$CONFIG_FILE"
fi

# GPS serial port will be /dev/serial0 (alias for ttyAMA0 on Pi 3 with bt swap)
ok "Serial configured — GPS will appear on /dev/serial0"

# -----------------------------------------------------------------------------
# 3. Adafruit 2.8" PiTFT (ILI9341, resistive touch)
# -----------------------------------------------------------------------------
info "Configuring Adafruit 2.8\" PiTFT display..."

# ILI9341 240x320 resistive touch, landscape rotation
if ! grep -q "pitft28-resistive" "$CONFIG_FILE"; then
  echo "" >> "$CONFIG_FILE"
  echo "# Adafruit 2.8\" PiTFT (ILI9341, resistive)" >> "$CONFIG_FILE"
  echo "dtoverlay=pitft28-resistive,rotate=90,speed=64000000,fps=30" >> "$CONFIG_FILE"
fi

# Ensure SPI is enabled (TFT uses it)
raspi-config nonint do_spi 0

# Blacklist the stmpe touch driver if it causes issues (optional)
# The pitft28-resistive overlay handles touch via stmpe610

ok "PiTFT configured — display will be /dev/fb1 at 320x240 (landscape)"

# -----------------------------------------------------------------------------
# 4. Install build tools and RTKLIB
# -----------------------------------------------------------------------------
info "Installing build dependencies..."
apt-get install -y -qq \
  build-essential cmake git python3-pip python3-venv \
  libncurses-dev zlib1g-dev \
  fonts-dejavu-core \
  2>/dev/null
ok "Build tools installed"

info "Building RTKLIB from source..."
RTKLIB_DIR="/opt/rtklib"
if [[ ! -d "$RTKLIB_DIR" ]]; then
  git clone --depth 1 https://github.com/rtklibexplorer/RTKLIB.git "$RTKLIB_DIR" 2>/dev/null || \
  git clone --depth 1 https://github.com/tomojitakasu/RTKLIB.git "$RTKLIB_DIR" 2>/dev/null
fi

cd "$RTKLIB_DIR/app/consapp"
for app in convbin rnx2rtkp rtkrcv; do
  if [[ ! -d "$app" ]]; then
    continue
  fi
  info "  Building $app..."
  cd "$app"
  make -j4 clean all 2>/dev/null
  cp -f "$app" /usr/local/bin/ 2>/dev/null || true
  cd ..
done

# Also build the demo5 fork if available (better RTCM3 MSM support)
if [[ -x /usr/local/bin/convbin ]]; then
  ok "RTKLIB installed: convbin, rnx2rtkp, rtkrcv"
else
  warn "RTKLIB build incomplete — some tools may be missing"
fi

# -----------------------------------------------------------------------------
# 5. Python virtual environment
# -----------------------------------------------------------------------------
info "Setting up Python environment..."
VENV_DIR="$PI_HOME/.venv/tep-pi"
if [[ ! -d "$VENV_DIR" ]]; then
  sudo -u "$PI_USER" python3 -m venv "$VENV_DIR"
fi

# Install Pi-specific requirements
info "Installing Python packages (this takes a while on Pi 3)..."
sudo -u "$PI_USER" "$VENV_DIR/bin/pip" install --upgrade pip wheel setuptools -q
sudo -u "$PI_USER" "$VENV_DIR/bin/pip" install -r "$REPO_DIR/rpi/requirements_pi.txt" -q

# Also install the main project requirements (lightweight subset)
sudo -u "$PI_USER" "$VENV_DIR/bin/pip" install numpy scipy -q

ok "Python environment ready at $VENV_DIR"

# -----------------------------------------------------------------------------
# 6. Framebuffer permissions
# -----------------------------------------------------------------------------
info "Setting framebuffer permissions..."
usermod -aG video,spi,gpio,dialout,tty "$PI_USER" 2>/dev/null || true

# Ensure /dev/fb1 is accessible
if [[ -e /dev/fb1 ]]; then
  chmod 666 /dev/fb1
fi

ok "Permissions set"

# -----------------------------------------------------------------------------
# 7. systemd service
# -----------------------------------------------------------------------------
info "Creating systemd service..."
SERVICE_FILE="/etc/systemd/system/tep-pi.service"
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=TEP-PPP Raspberry Pi Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$PI_USER
WorkingDirectory=$REPO_DIR
ExecStart=$VENV_DIR/bin/python $REPO_DIR/rpi/tep_pi_service.py
Environment=SDL_FBDEV=/dev/fb1
Environment=SDL_MOUSEDEV=/dev/input/event0
Environment=SDL_MOUSEDRV=IMPS2
Environment=PYTHONPATH=$REPO_DIR
Environment=GPS_SERIAL_PORT=/dev/serial0
Environment=GPS_BAUDRATE=9600
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable tep-pi.service
ok "Service installed: tep-pi.service (starts on boot)"

# -----------------------------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------------------------
echo ""
printf "${BOLD}${GREEN}========================================${NC}\n"
printf "${BOLD}${GREEN} TEP-PPP Pi Setup Complete${NC}\n"
printf "${BOLD}${GREEN}========================================${NC}\n"
echo ""
printf "  ${BOLD}Next steps:${NC}\n"
printf "  1. Reboot: ${CYAN}sudo reboot${NC}\n"
printf "  2. After reboot, the TFT should show the TEP display\n"
printf "  3. Check service: ${CYAN}systemctl status tep-pi${NC}\n"
printf "  4. View logs:     ${CYAN}journalctl -u tep-pi -f${NC}\n"
printf "  5. Stop service:  ${CYAN}sudo systemctl stop tep-pi${NC}\n"
printf "  6. Start manual:  ${CYAN}cd $REPO_DIR && $VENV_DIR/bin/python rpi/tep_pi_service.py${NC}\n"
echo ""
printf "  ${BOLD}GPS wiring (LEA-6H to Pi GPIO):${NC}\n"
printf "    GPS TX  -> Pi RX  (GPIO 15, pin 10)\n"
printf "    GPS RX  -> Pi TX  (GPIO 14, pin  8)\n"
printf "    GPS VCC -> 3.3V   (pin 1 or 17)\n"
printf "    GPS GND -> GND    (pin 6 or 9)\n"
echo ""
printf "  ${BOLD}TFT wiring (PiTFT sits on GPIO header):${NC}\n"
printf "    Just seat the PiTFT on the 40-pin header.\n"
printf "    GPS wires connect to the PiTFT's pass-through pins.\n"
echo ""
