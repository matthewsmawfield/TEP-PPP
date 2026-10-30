"""Guards for OS Net station positions.

ARDL was absent from both position sources -- the 10-entry OS_NET_STATIONS
table and the DMS station-info endpoint -- so it silently kept its (0.0, 0.0)
default and sat on Null Island, ~5900 km from the rest of the UK network.  That
corrupted the kriging geometry for every leave-one-out case it entered, and
inflated the apparent geometry contribution.  Positions now come from each
station's own RINEX header.
"""
import io
import math
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lib.osnet_client import _header_lla          # noqa: E402
from lib.ntrip_client import OS_NET_STATIONS      # noqa: E402

# EASN (Easington), true position 53.648991 N, 0.116435 E
_X, _Y, _Z = 3788778.8796, 7699.4690, 5113731.9448
_HEADER = (
    "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n"
    "TPP 5.3.0                               20260907 110000 UTC PGM / RUN BY / DATE\n"
    "EASN                                                        MARKER NAME\n"
    "   3788778.8796     7699.4690  5113731.9448                    APPROX POSITION XYZ\n"
    "                                                            END OF HEADER\n"
)


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("EASN_2026_250_11.rnx", text)
    return buf.getvalue()


def test_header_lla_recovers_station_position():
    lat, lon = _header_lla(_zip(_HEADER))
    assert lat == pytest.approx(53.649, abs=0.01)
    assert lon == pytest.approx(0.1164, abs=0.01)


def test_header_lla_rejects_null_island():
    """A zeroed APPROX POSITION must not be accepted as a real fix."""
    zeroed = _HEADER.replace(f"{_X:13.4f}{_Y:14.4f}{_Z:14.4f}",
                             f"{0.0:13.4f}{0.0:14.4f}{0.0:14.4f}")
    assert _header_lla(_zip(zeroed)) is None


def test_header_lla_returns_none_without_position():
    stripped = "\n".join(L for L in _HEADER.splitlines()
                         if "APPROX POSITION" not in L) + "\n"
    assert _header_lla(_zip(stripped)) is None


def test_no_station_in_table_sits_on_null_island():
    for name, (lat, lon, _dist) in OS_NET_STATIONS.items():
        assert abs(lat) + abs(lon) > 0.01, f"{name} is on Null Island"
        assert 49.0 < lat < 61.0, f"{name} latitude {lat} is outside the UK"


def test_table_stations_are_mutually_within_the_uk():
    """No pair may exceed the ~1000 km extent of the UK network."""
    def gc(a, b):
        p1, p2 = math.radians(a[0]), math.radians(b[0])
        dl = math.radians(b[1] - a[1])
        v = (math.sin(p1) * math.sin(p2)
             + math.cos(p1) * math.cos(p2) * math.cos(dl))
        return 6371.0 * math.acos(max(-1.0, min(1.0, v)))

    items = list(OS_NET_STATIONS.items())
    for i, (na, a) in enumerate(items):
        for nb, b in items[i + 1:]:
            d = gc(a, b)
            assert d < 1000.0, f"{na}-{nb} baseline {d:.0f} km exceeds the UK"
