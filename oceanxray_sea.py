"""oceanxray_sea.py -- wave height, wave period and sea-surface temperature for June 2021 (ERA5).

Needs one data file next to this script: oceanxray_sea_data.bin (about 0.9 MB).
Contents: ERA5 hourly data on single levels (Copernicus Climate Change Service), June 2021, 4-24 N,
68-94 E: significant wave height and mean wave period (0.5 degree grid) and sea-surface temperature
(0.25 degree grid), kept every 3 hours. Values for a scene are interpolated in space and time.
The file is looked for in the same folder as this script, the current folder, /kaggle/working and
under /kaggle/input; set OCEANXRAY_SEA_DATA to a full path to override. If it is missing,
sea_state() returns None and the dashboard falls back to its other sources.

    sea_state(lat, lon, when_utc) -> dict(wave_h, wave_t, sst_c, hour_utc, note) or None
Contains modified Copernicus Climate Change Service information (ERA5, Hersbach et al. 2023).
"""
import glob
import json
import os
import zlib
from datetime import datetime, timezone

import numpy as np

_FILE = "oceanxray_sea_data.bin"
_STORE = None


def _find():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    cands = [os.environ.get("OCEANXRAY_SEA_DATA", ""), os.path.join(here, _FILE),
             os.path.join(os.getcwd(), _FILE), os.path.join("/kaggle/working", _FILE)]
    cands += glob.glob(os.path.join("/kaggle/input", "**", _FILE), recursive=True)
    return next((c for c in cands if c and os.path.isfile(c)), None)


def _load():
    global _STORE
    if _STORE is None:
        path = _find()
        if path is None:
            _STORE = False
        else:
            raw = open(path, "rb").read()
            n = int.from_bytes(raw[:4], "little")
            meta = json.loads(raw[4:4 + n])
            pos, arrs = 4 + n, {}
            for key, size in zip(("swh", "mwp", "sst"), meta["sizes"]):
                shape = (meta["nt"], *(meta["sst"]["shape"] if key == "sst" else meta["wave"]["shape"]))
                d = np.frombuffer(zlib.decompress(raw[pos:pos + size]), "<i4").reshape(shape)
                q = np.cumsum(d, axis=0)
                a = q.astype(np.float32) / meta["scale"]
                a[q == -32768] = np.nan
                arrs[key] = a
                pos += size
            _STORE = (meta, arrs)
    return _STORE or None


def _spatial(a, spec, lat, lon, max_deg):
    """Bilinear value from the valid neighbours of one time slice; else nearest valid cell within max_deg.
    -> (value or None, used_nearest_cell)"""
    step, la0, lo0 = spec["step"], spec["lat0"], spec["lon0"]
    ny, nx = a.shape
    fy, fx = (la0 - lat) / step, (lon - lo0) / step
    if not (0 <= fy <= ny - 1 and 0 <= fx <= nx - 1):
        return None, False
    y0, x0 = int(np.floor(fy)), int(np.floor(fx))
    y1, x1 = min(y0 + 1, ny - 1), min(x0 + 1, nx - 1)
    wy, wx = fy - y0, fx - x0
    tot = acc = 0.0
    for y, x, w in ((y0, x0, (1 - wy) * (1 - wx)), (y0, x1, (1 - wy) * wx),
                    (y1, x0, wy * (1 - wx)), (y1, x1, wy * wx)):
        v = a[y, x]
        if w > 0 and not np.isnan(v):
            tot += w
            acc += w * v
    if tot >= 0.999:
        return acc / tot, False
    r = int(np.ceil(max_deg / step))
    best = None
    for y in range(max(0, y0 - r), min(ny, y0 + r + 2)):
        for x in range(max(0, x0 - r), min(nx, x0 + r + 2)):
            if not np.isnan(a[y, x]):
                dd = ((y - fy) * step) ** 2 + ((x - fx) * step) ** 2
                if dd <= max_deg ** 2 and (best is None or dd < best[0]):
                    best = (dd, a[y, x])
    return (float(best[1]), True) if best else (None, False)


def sea_state(lat, lon, when_utc, max_deg=1.0):
    st = _load()
    if st is None or when_utc is None:
        return None
    meta, arrs = st
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    t0 = datetime.fromisoformat(meta["t0"]).replace(tzinfo=timezone.utc)
    f = (when_utc - t0).total_seconds() / 3600.0 / meta["dt_h"]
    if not (0 <= f <= meta["nt"] - 1):
        return None
    i0 = int(np.floor(f)); i1 = min(i0 + 1, meta["nt"] - 1); w = f - i0
    out, nearest = {}, False
    for key, spec, name, md in (("swh", "wave", "wave_h", max_deg), ("mwp", "wave", "wave_t", max_deg),
                                ("sst", "sst", "sst_c", 0.5)):
        v0, n0 = _spatial(arrs[key][i0], meta[spec], lat, lon, md)
        v1, n1 = _spatial(arrs[key][i1], meta[spec], lat, lon, md)
        out[name] = None if v0 is None or v1 is None else float((1 - w) * v0 + w * v1)
        nearest = nearest or n0 or n1
    if all(out[k] is None for k in out):
        return None
    out["hour_utc"] = when_utc.strftime("%Y-%m-%dT%H:%M")
    out["nearest_cell"] = nearest
    out["note"] = ("ERA5 reanalysis (waves about 55 km grid, sea temperature about 28 km grid), interpolated to the "
                   "scene time" + ("; the tile is near the coast, so the nearest open-water value is shown" if nearest else ""))
    return out


def selftest():
    """Quick checks; prints a line per case."""
    d = sea_state(15.0, 85.0, datetime(2021, 6, 15, 12, 0, tzinfo=timezone.utc))
    assert d and 0.3 < d["wave_h"] < 6 and 3 < d["wave_t"] < 14 and 26 < d["sst_c"] < 32, d
    assert sea_state(15.0, 85.0, datetime(2021, 8, 1, tzinfo=timezone.utc)) is None
    assert sea_state(40.0, 85.0, datetime(2021, 6, 15, tzinfo=timezone.utc)) is None
    print("oceanxray_sea selftest OK", d)


if __name__ == "__main__":
    selftest()