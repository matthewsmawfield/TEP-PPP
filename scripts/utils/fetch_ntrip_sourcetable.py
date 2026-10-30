#!/usr/bin/env python3
"""
NTRIP Sourcetable Fetcher
=========================
Connects to an NTRIP caster and retrieves the sourcetable to identify active mountpoints.
Useful for finding valid streams on public casters like RTK2go.

Usage:
    python scripts/utils/fetch_ntrip_sourcetable.py --caster rtk2go.com --port 2101
"""

import socket
import argparse
import sys

def fetch_sourcetable(caster, port):
    print(f"Fetching sourcetable from {caster}:{port}...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((caster, port))
        
        # Standard NTRIP request for sourcetable (GET /)
        req = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {caster}\r\n"
            "Ntrip-Version: Ntrip/2.0\r\n"
            "User-Agent: NTRIP TEP-PPP-Fetcher/1.0\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        
        s.sendall(req.encode())
        
        response = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            response += chunk
            
        s.close()
        
        text = response.decode(errors='ignore')
        
        # Split header and body
        if "\r\n\r\n" in text:
            header, body = text.split("\r\n\r\n", 1)
        else:
            header = text
            body = ""
            
        print("\n--- CASTER RESPONSE HEADER ---")
        print(header)
        print("------------------------------\n")
        
        streams = []
        for line in body.splitlines():
            if line.startswith("STR;"):
                parts = line.split(";")
                if len(parts) > 1:
                    streams.append(parts[1])
                    
        print(f"Found {len(streams)} streams.")
        if streams:
            print("First 10 streams found:")
            for mp in streams[:10]:
                print(f" - {mp}")
                
        return streams

    except Exception as e:
        print(f"Error fetching sourcetable: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caster", default="rtk2go.com", help="Caster hostname")
    parser.add_argument("--port", type=int, default=2101, help="Caster port")
    args = parser.parse_args()
    
    fetch_sourcetable(args.caster, args.port)
