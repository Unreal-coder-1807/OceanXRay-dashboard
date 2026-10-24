"""oceanxray_bathy.py -- sea-floor depth for any point in the study area (GEBCO).

Needs one data file next to this script: oceanxray_bathy_grid.bin (0.05 degree, about 5.5 km,
GEBCO grid for 4-24 N, 68-94 E; zlib-compressed int16, 5 m steps; about 220 KB).
It is looked for in the same folder as this file, the current folder, /kaggle/working, and under
/kaggle/input; set the environment variable OCEANXRAY_BATHY_GRID to a full path to override.
GEBCO Compilation Group (2026) GEBCO Grid. https://www.gebco.net  (terms: free to use with attribution)

    depth_at(lat, lon)        -> metres below sea level (negative = land) or None outside the grid
    describe_seafloor(lat, lon) -> dict(depth_m, zone, slope_m_per_km, shelf_break_km, text)
"""
import glob
import math
import os
import zlib

import numpy as np

_N, _M, _LA0, _LO0, _STEP = 400, 520, 4.025, 68.025, 0.05
_GRID = None
_FILE = "oceanxray_bathy_grid.bin"


def _find_grid_file():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    cands = [os.environ.get("OCEANXRAY_BATHY_GRID", ""), os.path.join(here, _FILE),
             os.path.join(os.getcwd(), _FILE), os.path.join("/kaggle/working", _FILE)]
    cands += glob.glob(os.path.join("/kaggle/input", "**", _FILE), recursive=True)
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def _grid():
    """Loads the grid once; returns None (and the depth panel simply stays hidden) if the file is missing."""
    global _GRID
    if _GRID is None:
        path = _find_grid_file()
        if path is None:
            return None
        with open(path, "rb") as f:
            q = np.frombuffer(zlib.decompress(f.read()), dtype=np.int16).reshape(_N, _M)
        _GRID = q.astype(np.float32) * 5.0          # metres, negative = below sea level
    return _GRID


def _ij(lat, lon):
    return (lat - _LA0) / _STEP, (lon - _LO0) / _STEP


def depth_at(lat, lon):
    g = _grid()
    if g is None:
        return None
    fi, fj = _ij(lat, lon)
    if not (0 <= fi < _N - 1 and 0 <= fj < _M - 1):
        return None
    i, j = int(fi), int(fj)
    a, b = fi - i, fj - j
    z = (g[i, j] * (1 - a) * (1 - b) + g[i + 1, j] * a * (1 - b)
         + g[i, j + 1] * (1 - a) * b + g[i + 1, j + 1] * a * b)
    return float(-z)


def _zone(d):
    if d <= 0:
        return "land"
    if d < 50:
        return "very shallow shelf"
    if d < 200:
        return "continental shelf"
    if d < 1000:
        return "upper slope"
    if d < 3000:
        return "deep slope"
    return "deep ocean"


def describe_seafloor(lat, lon, radius_km=15.0, search_km=150.0):
    d = depth_at(lat, lon)
    if d is None:
        return None                    # outside the grid, or the .bin file was not found
    g = _grid()
    fi, fj = _ij(lat, lon)
    ci, cj = int(round(fi)), int(round(fj))
    kmpx = _STEP * 111.0
    r = max(1, int(radius_km / kmpx))
    sl = g[max(ci - r, 1):ci + r + 1, max(cj - r, 1):cj + r + 1]
    gy, gx = np.gradient(sl, kmpx)
    slope = float(np.nanmax(np.hypot(gx, gy))) if sl.size else 0.0
    # distance to the 200 m contour (shelf break)
    R = int(search_km / kmpx)
    i0, i1 = max(ci - R, 0), min(ci + R + 1, _N)
    j0, j1 = max(cj - R, 0), min(cj + R + 1, _M)
    win = -g[i0:i1, j0:j1]
    near = np.abs(win - 200.0) < 40.0
    sb = None
    if near.any():
        ii, jj = np.nonzero(near)
        dy = (ii + i0 - fi) * kmpx
        dx = (jj + j0 - fj) * kmpx * math.cos(math.radians(lat))
        sb = float(np.min(np.hypot(dx, dy)))
    zone = _zone(d)
    if d <= 0:
        text = "on land in the depth grid"
    else:
        text = f"sea floor about {d:,.0f} m deep ({zone})"
        if sb is not None and sb <= 60 and d >= 20:
            text += f", about {sb:.0f} km from the shelf edge (200 m depth)"
        if slope >= 40:
            text += f"; steep sea-floor slope nearby (up to {slope:.0f} m per km)"
    return {"depth_m": d, "zone": zone, "slope_m_per_km": slope, "shelf_break_km": sb, "text": text}


def selftest():
    assert _grid() is not None, f"{_FILE} not found next to this script"
    assert abs(depth_at(21.5, 90.8) - 20) < 40                      # shallow northern Bay of Bengal shelf
    assert depth_at(19.7, 89.2) > 800                               # Swatch of No Ground, canyon head area
    assert depth_at(23.0, 80.0) is None or depth_at(23.0, 80.0) < 0  # land (India)
    assert depth_at(0.0, 80.0) is None                              # outside grid
    s = describe_seafloor(20.6, 89.1)
    assert s and s["depth_m"] > 100, s
    print("oceanxray_bathy self-test: all checks passed")
    for p in ((21.5, 90.8), (20.6, 89.1), (8.0, 93.5), (13.0, 85.0)):
        print(p, describe_seafloor(*p)["text"])



if __name__ == "__main__":
    selftest()