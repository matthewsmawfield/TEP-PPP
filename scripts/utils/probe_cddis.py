import requests
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.utils.logger import TEPLogger, print_status

def main():
    # Auth
    import netrc
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if auth:
            user, _, pwd = auth
    except:
        user = os.getenv("CDDIS_USER")
        pwd = os.getenv("CDDIS_PASS")
    
    if not user:
        print("No auth")
        return

    session = requests.Session()
    session.auth = (user, pwd)
    
    # Try listing 2024/100 troposphere products
    url = "https://cddis.nasa.gov/archive/gnss/products/troposphere/zpd/2024/100/"
    # Note: CDDIS might not support directory listing via simple GET if index is disabled, but usually it works or we need to guess filenames.
    # However, CDDIS moved to HTTPS and Earthdata login.
    # Let's try to hit a likely file.
    
    candidates = [
        "IGS0OPSFIN_20241000000_01D_05M_TRO.TRO.gz",
        "zimm1000.24zpd.gz",
        "zimm1000.24z.gz"
    ]
    
    for c in candidates:
        full_url = url + c
        print(f"HEAD {full_url}")
        r = session.head(full_url)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print(f"FOUND: {c}")

if __name__ == "__main__":
    main()
