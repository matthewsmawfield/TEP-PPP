#!/usr/bin/env python3
"""Filter satellites out of a RINEX 3 observation file.

Writes a copy of the obs file with all observation records for the
listed satellites removed and each epoch's satellite count rewritten.
Only flag-0/1 epoch records are modified; special-event epoch records
(flag>=2) and the header are passed through verbatim.  Used by the
PPP-AR multi-start to generate an exclusion-decorrelated draw.
"""
from __future__ import annotations

from pathlib import Path


def filter_obs_sats(obs_path: Path, out_path: Path,
                    exclude: set[str]) -> int:
    """Copy obs_path to out_path without the excluded satellites.

    Returns the number of observation records dropped.  0 means no
    matching records were found (the sats were absent).
    """
    dropped = 0
    end_of_header = False
    with open(obs_path, "r", errors="ignore") as fin, \
            open(out_path, "w", errors="ignore") as fout:
        line = fin.readline()
        while line:
            if not end_of_header:
                fout.write(line)
                if "END OF HEADER" in line:
                    end_of_header = True
                line = fin.readline()
                continue
            if not line.startswith(">"):
                # Stray line between epochs — pass through verbatim.
                fout.write(line)
                line = fin.readline()
                continue
            # Epoch line: "> yyyy mm dd hh mm ss.sssssss flag nrec"
            p = line.split()
            try:
                flag = int(p[7])
                nrec = int(p[8])
            except (IndexError, ValueError):
                fout.write(line)
                line = fin.readline()
                continue
            if flag > 1:
                # Special event: nrec is a record count of non-obs
                # records — pass through verbatim.
                fout.write(line)
                line = fin.readline()
                continue
            recs = []
            for _ in range(nrec):
                rec = fin.readline()
                if not rec:
                    break
                recs.append(rec)
            kept = [r for r in recs if r[:3].strip() not in exclude]
            dropped += len(recs) - len(kept)
            if kept:
                # Rewrite only the record-count field (cols 33-35,
                # left-justified in this file's layout) — preserving
                # the rest of the epoch line byte-for-byte.
                fout.write(f"{line[:33]}{len(kept):<3}{line[36:]}")
                for r in kept:
                    fout.write(r)
            line = fin.readline()
    return dropped
