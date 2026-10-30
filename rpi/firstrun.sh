#!/bin/bash
set -e

# =============================================================================
#  TEP-PPP Raspberry Pi first-boot configuration
#  - Hostname:   tep-pi
#  - SSH:        enabled
#  - User:       pi (with sudo, dialout, video, spi, gpio, tty groups)
#  - Wi-Fi:      configured via WIFI_SSID / WIFI_PSK environment variables
# =============================================================================

WIFI_SSID="${WIFI_SSID:-YOUR_WIFI_SSID}"
WIFI_PSK="${WIFI_PSK:-YOUR_WIFI_PASSWORD}"
PI_PASSWORD="${PI_PASSWORD:-CHANGE_ME}"

# --- Hostname -----------------------------------------------------------------
echo "tep-pi" > /etc/hostname
sed -i "s/^127\.0\.1\.1.*/127.0.1.1\ttep-pi/" /etc/hosts 2>/dev/null || true

# --- SSH ----------------------------------------------------------------------
systemctl enable ssh

# --- User (Bookworm has no default pi user) -----------------------------------
if ! id -u pi >/dev/null 2>&1; then
  useradd -m -s /bin/bash -G sudo,dialout,video,spi,gpio,tty pi
fi
echo "pi:${PI_PASSWORD}" | chpasswd

# --- Wi-Fi via NetworkManager -------------------------------------------------
mkdir -p /etc/NetworkManager/system-connections
cat > /etc/NetworkManager/system-connections/wifi.nmconnection << WIFIEOF
[connection]
id=${WIFI_SSID}
type=wifi
autoconnect=true
[wifi]
mode=infrastructure
ssid=${WIFI_SSID}
[wifi-security]
auth-alg=open
key-mgmt=wpa-psk
psk=${WIFI_PSK}
[ipv4]
method=auto
[ipv6]
addr-gen-mode=default
method=auto
WIFIEOF
chmod 600 /etc/NetworkManager/system-connections/wifi.nmconnection

# --- Clean up first-run artefacts ---------------------------------------------
rm -f /usr/lib/raspberrypi-sys-mods/firstboot.sh
rm -f /etc/systemd/system/firstboot.service
systemctl daemon-reload

echo "First-run configuration complete."
