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
    
    gps_week = 2309 # 2024 DOY 100 is roughly week 2309 (April 9 2024)
    # 2024-04-09. GPS Epoch 1980-01-06.
    # Days = (2024*365 + ...) 
    # Let's trust the script logic or calculate it.
    # 2024 is leap. 
    # April 9 is DOY 100.
    
    url_base = f"https://cddis.nasa.gov/archive/gnss/products/{gps_week}/"
    
    # Common MGEX names for 2024 (Long filenames)
    # CODE: COD0MGXFIN_20241000000_01D_05M_ORB.SP3.gz
    # GFZ:  GFZ0MGXRAP_20241000000_01D_05M_ORB.SP3.gz (RAPid) or FIN
    # GRG:  GRG0MGXfin...
    # JPL:  JPL0MGXFIN...
    
    prefixes = ["COD", "GFZ", "GRG", "JPL", "IAC", "SHA"]
    types = ["MGXFIN", "MGXRAP", "OPSFIN", "OPSRAP"]
    
    for p in prefixes:
        for t in types:
            fname = f"{p}0{t}_20241000000_01D_05M_ORB.SP3.gz"
            full_url = url_base + fname
            print(f"Checking {fname}...", end=" ", flush=True)
            try:
                r = session.head(full_url, timeout=5)
                if r.status_code == 200:
                    print("FOUND")
                else:
                    print(f"{r.status_code}")
            except:
                print("Error")

if __name__ == "__main__":
    main()
