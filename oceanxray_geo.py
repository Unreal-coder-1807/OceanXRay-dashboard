"""
OceanXRay -- shared geolocation helpers for internal-wave tiles.

Used by infer_iw_on_real_scenes.py (batch run over our scenes) and by
iw_dashboard.py (geolocating a single uploaded tile). Keeping one copy
means the dashboard map and the detections file can never disagree about
where a pixel is.

What it does
------------
1. Parses tile filenames written by convert_scenes_to_images.py:
       {scene_id}_tile{idx}_row{row_start}_col{col_start}.png
   The row/col are the tile's top-left corner in the ORIGINAL scene's
   pixels. Each tile covers 2,500 x 2,500 scene pixels (25 km at 10 m)
   and was resized to 1,024 x 1,024, so 1 tile pixel = 2.44 scene pixels.
2. Converts scene pixels to longitude/latitude with the same GCP-based
   affine transform the rest of the project uses (rasterio's from_gcps,
   a least-squares fit to the scene's ground-control points).
3. Stores that transform as 6 numbers per scene in a small JSON table,
   so the dashboard can geolocate without the multi-GB .SAFE folders.
4. Land / coastline checks against land_ocean_mask.npz.

Build the table (once, wherever the .SAFE folders are, e.g. Kaggle):
    set SAFE_ROOT below, then run this file (works as a script or in a
    notebook cell via %run oceanxray_geo.py). Writes scene_geo_table.json.
    Copy that JSON next to iw_dashboard.py.

Accuracy note: one affine transform per scene is an approximation of the
true SAR geometry (GCP grid). It is the same approximation every map in
this project already uses; treat positions as approximate (sub-km to
~1 km scale), not survey-grade.
"""

import json
import os
import re

import numpy as np

# ---- EDIT THESE (only needed when building the table) ----
SAFE_ROOT = "/kaggle/input/datasets/andrejkarpathy/sentinel-1-june-2021-raw-safe"                      # folder containing the *.SAFE scene folders
GEO_TABLE_PATH = "scene_geo_table.json"
# ----------------------------------------------------------

TILE_SIZE_KM = 25                     # must match convert_scenes_to_images.py
OUTPUT_PIXELS = 1024                  # must match convert_scenes_to_images.py
NATIVE_PX_M = 10.0                    # Sentinel-1 IW GRD pixel spacing
NATIVE_TILE_PX = int(TILE_SIZE_KM * 1000 / NATIVE_PX_M)   # 2500
TILE_TO_SCENE = NATIVE_TILE_PX / OUTPUT_PIXELS            # 2.44 scene px per tile px
COAST_BUFFER_KM = 1.0                 # same as tier1_fft_backtrack.py

# Region of interest used across the project (lon_min, lat_min, lon_max, lat_max)
ROI_BOUNDS = (69.0, 4.0, 96.0, 24.0)

TILE_FILENAME_RE = re.compile(r"^(.+)_tile(\d+)_row(\d+)_col(\d+)\.png$", re.IGNORECASE)
_SCENE_TIME_RE = re.compile(r"_(\d{8})T(\d{6})_")


# ----------------------------------------------------------------------
# Filenames
# ----------------------------------------------------------------------
def parse_tile_filename(filename):
    """-> (scene_id, tile_idx, row_start, col_start), or None if the name
    doesn't follow the pipeline's pattern (e.g. a benchmark image)."""
    m = TILE_FILENAME_RE.match(os.path.basename(filename))
    if not m:
        return None
    scene_id, tile_idx, row_start, col_start = m.groups()
    return scene_id, int(tile_idx), int(row_start), int(col_start)


def scene_datetime_str(scene_id):
    """'S1A_IW_GRDH_1SDV_20210626T120406_...' -> '2021-06-26 12:04 UTC'."""
    m = _SCENE_TIME_RE.search(scene_id)
    if not m:
        return None
    d, t = m.groups()
    return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]} UTC"


# ----------------------------------------------------------------------
# Transform table
# ----------------------------------------------------------------------
class SceneTransform:
    """Scene pixel (x = column, y = row) -> (lon, lat), from 6 affine numbers
    in rasterio's order: lon = a*x + b*y + c ; lat = d*x + e*y + f."""

    def __init__(self, coeffs, width=None, height=None):
        self.a, self.b, self.c, self.d, self.e, self.f = [float(v) for v in coeffs]
        self.width, self.height = width, height

    def __call__(self, x, y):
        return self.a * x + self.b * y + self.c, self.d * x + self.e * y + self.f

    # Same calling style as rasterio's Affine: T * (x, y)
    def __mul__(self, xy):
        return self(*xy)

    def footprint(self):
        """Scene corners as [[lon, lat], ...] (needs width/height)."""
        if self.width is None or self.height is None:
            return None
        w, h = self.width, self.height
        return [list(self(x, y)) for x, y in ((0, 0), (w, 0), (w, h), (0, h))]


def build_geo_table(safe_root=SAFE_ROOT, out_path=GEO_TABLE_PATH, polarization="vv"):
    """Read every *.SAFE under safe_root, fit the GCP affine, save JSON."""
    import glob
    import rasterio
    from rasterio.transform import from_gcps

    table, failed = {}, []
    safes = sorted(e for e in os.listdir(safe_root) if e.endswith(".SAFE"))
    print(f"Found {len(safes)} .SAFE folders under '{safe_root}'")
    for entry in safes:
        scene_id = entry[:-5]
        hits = glob.glob(os.path.join(safe_root, entry, "measurement", f"*{polarization}*.tiff"))
        if not hits:
            hits = glob.glob(os.path.join(safe_root, entry, "**", "measurement",
                                          f"*{polarization}*.tiff"), recursive=True)
        if not hits:
            failed.append((scene_id, "no measurement tiff"))
            continue
        try:
            with rasterio.open(hits[0]) as src:
                gcps, _ = src.get_gcps()
                if not gcps:
                    raise ValueError("no GCPs")
                A = from_gcps(gcps)
                table[scene_id] = {
                    "affine": [A.a, A.b, A.c, A.d, A.e, A.f],
                    "width": src.width, "height": src.height,
                    "time": scene_datetime_str(scene_id),
                }
        except Exception as e:                      # keep going; report at the end
            failed.append((scene_id, str(e)))
    with open(out_path, "w") as f:
        json.dump(table, f)
    print(f"Saved {len(table)} scene transforms to {out_path}")
    for sid, why in failed:
        print(f"  SKIPPED {sid}: {why}")
    return table


def load_geo_table(path=GEO_TABLE_PATH):
    """-> {scene_id: SceneTransform}, plus the raw dict for metadata."""
    with open(path) as f:
        raw = json.load(f)
    return {sid: SceneTransform(v["affine"], v.get("width"), v.get("height"))
            for sid, v in raw.items()}, raw


# ----------------------------------------------------------------------
# Tile / box geometry
# ----------------------------------------------------------------------
def tile_px_to_scene(row_start, col_start, x, y):
    """Resized-tile pixel (x, y) -> scene pixel (x, y)."""
    return col_start + x * TILE_TO_SCENE, row_start + y * TILE_TO_SCENE


def tile_footprint(T, row_start, col_start):
    """Tile corners as [[lon, lat]] x 4 (TL, TR, BR, BL)."""
    n = OUTPUT_PIXELS
    return [list(T(*tile_px_to_scene(row_start, col_start, x, y)))
            for x, y in ((0, 0), (n, 0), (n, n), (0, n))]


def box_polygon(T, row_start, col_start, box):
    """Detection box [x1, y1, x2, y2] in resized-tile pixels -> 4 corners [[lon, lat]]."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return [list(T(*tile_px_to_scene(row_start, col_start, x, y)))
            for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]


def box_center(T, row_start, col_start, box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return T(*tile_px_to_scene(row_start, col_start, (x1 + x2) / 2, (y1 + y2) / 2))


# ----------------------------------------------------------------------
# Land / coast
# ----------------------------------------------------------------------
def load_land_checker(path="land_ocean_mask.npz"):
    """-> is_on_land(lat, lon). Same lookup as the rest of the project."""
    d = np.load(path)
    mask, mlat, mlon = d["mask"], d["lat"], d["lon"]
    dlat, dlon = mlat[1] - mlat[0], mlon[1] - mlon[0]
    lat_lo, lat_hi, lon_lo, lon_hi = mlat.min(), mlat.max(), mlon.min(), mlon.max()

    def is_on_land(lat, lon):
        lat = max(lat_lo, min(lat_hi, lat))
        lon = max(lon_lo, min(lon_hi, lon))
        i = min(int((lat - mlat[0]) / dlat), mask.shape[0] - 1)
        j = min(int((lon - mlon[0]) / dlon), mask.shape[1] - 1)
        return not bool(mask[i, j])          # mask True = ocean
    return is_on_land


def box_touches_land(T, row_start, col_start, box, is_on_land,
                     buffer_km=COAST_BUFFER_KM, n=9):
    """True if any point on an n x n grid over the box, grown by buffer_km,
    is on land. Same rule as tier1_fft_backtrack.py's coastline check."""
    x1, y1, x2, y2 = [float(v) for v in box]
    buf = buffer_km * 1000 / NATIVE_PX_M                       # scene pixels
    sx1, sy1 = tile_px_to_scene(row_start, col_start, x1, y1)
    sx2, sy2 = tile_px_to_scene(row_start, col_start, x2, y2)
    for sy in np.linspace(sy1 - buf, sy2 + buf, n):
        for sx in np.linspace(sx1 - buf, sx2 + buf, n):
            lon, lat = T(sx, sy)
            if is_on_land(lat, lon):
                return True
    return False


# ----------------------------------------------------------------------
# Self-test (no data needed)
# ----------------------------------------------------------------------
def selftest():
    # A made-up scene: north-up, 10 m pixels, top-left at 90E, 20N
    deg_lon, deg_lat = 10 / 104_600, 10 / 110_700
    T = SceneTransform([deg_lon, 0, 90.0, 0, -deg_lat, 20.0], 25000, 16000)
    assert parse_tile_filename("S1A_IW_GRDH_1SDV_20210626T120406_X_tile0073_row12000_col2000.png") \
        == ("S1A_IW_GRDH_1SDV_20210626T120406_X", 73, 12000, 2000)
    assert parse_tile_filename("P0001.png") is None
    assert scene_datetime_str("S1A_IW_GRDH_1SDV_20210626T120406_X") == "2021-06-26 12:04 UTC"
    fp = tile_footprint(T, 0, 0)
    w_km = (fp[1][0] - fp[0][0]) * 104.6
    h_km = (fp[0][1] - fp[3][1]) * 110.7
    assert abs(w_km - 25) < 0.01 and abs(h_km - 25) < 0.01, (w_km, h_km)
    poly = box_polygon(T, 0, 0, [0, 0, 1024, 1024])
    assert np.allclose(poly, fp)
    lon, lat = box_center(T, 0, 0, [0, 0, 1024, 1024])
    assert abs(lon - (90 + 1250 * deg_lon)) < 1e-9 and abs(lat - (20 - 1250 * deg_lat)) < 1e-9
    assert T * (0, 0) == (90.0, 20.0)
    # land: everything west of 90.3E is land in this fake checker
    fake_land = lambda la, lo: lo < 90.3
    assert box_touches_land(T, 0, 0, [0, 0, 100, 100], fake_land)          # box at 90.00E
    assert not box_touches_land(T, 0, 6000, [0, 0, 100, 100], fake_land)   # box at ~90.57E
    # 1 km buffer: box whose west edge sits 0.5 km east of the coast touches land
    col_coast = 0.3 / deg_lon                          # scene px of 90.3E
    col_box = col_coast + 50                           # 500 m east of the coast
    tile_col = int(col_box)                            # put the box at tile x = 0
    assert box_touches_land(T, 0, tile_col, [0, 0, 10, 10], fake_land)
    assert not box_touches_land(T, 0, tile_col, [0, 0, 10, 10], fake_land, buffer_km=0.2)
    print("oceanxray_geo self-test: all checks passed")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        build_geo_table()