"""
OceanXRay -- place names and scene-time weather for the dashboard.

Two small helpers, one file:

1. PLACES (works offline). Turns coordinates into words:
       Andaman Sea · 38 km E of Little Andaman (India)
   The sea name comes from Natural Earth marine polygons; the reference
   point is the nearest named island coast or town. Open ocean has no
   street address, so "sea + nearest land" is the honest description.
   Data (public domain, Natural Earth: seas, towns, island coasts) lives in
   one small file next to this script: oceanxray_context_data.bin
   (zlib-compressed JSON, about 25 KB). It is looked for in the same folder
   as this file, the current folder, /kaggle/working and under /kaggle/input;
   set OCEANXRAY_CONTEXT_DATA to a full path to override. If it is missing,
   importing this module raises ImportError, so the dashboard falls back to
   plain coordinates.

2. CONDITIONS (needs internet). Wind, waves and sea-surface temperature at
   the scene's date and hour, from the Open-Meteo historical API (ERA5
   reanalysis -- the same family of data the project already uses). These
   are modelled values on a coarse grid (about 25 km), not measurements at
   the exact spot. fetch_conditions() never raises: it returns
   (values or None, error text or None) so the dashboard can fall back.

Run `python oceanxray_context.py selftest` to check the place lookup.
"""

import glob
import json
import math
import os
import zlib
from datetime import datetime, timedelta, timezone

import numpy as np

# ----------------------------------------------------------------------
# Places
# ----------------------------------------------------------------------
_D = None
_FILE = "oceanxray_context_data.bin"


def _find_data_file():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    cands = [os.environ.get("OCEANXRAY_CONTEXT_DATA", ""), os.path.join(here, _FILE),
             os.path.join(os.getcwd(), _FILE), os.path.join("/kaggle/working", _FILE)]
    cands += glob.glob(os.path.join("/kaggle/input", "**", _FILE), recursive=True)
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


_DATA_PATH = _find_data_file()
if _DATA_PATH is None:
    raise ImportError(f"{_FILE} not found: put it next to oceanxray_context.py "
                      "(or set OCEANXRAY_CONTEXT_DATA)")


def _data():
    global _D
    if _D is None:
        with open(_DATA_PATH, "rb") as f:
            raw = json.loads(zlib.decompress(f.read()))
        seas = []
        for s in raw["seas"]:
            rings = [np.asarray(r, dtype=float) for r in s["r"]]
            area = sum(abs(_shoelace(r)) for r in rings)
            seas.append((s["n"], rings, area))
        pts = raw["pts"]
        _D = {
            "seas": seas,
            "lon": np.array([p[0] for p in pts]), "lat": np.array([p[1] for p in pts]),
            "name": [p[2] for p in pts], "cc": [p[3] for p in pts], "kind": [p[4] for p in pts],
        }
    return _D


def _shoelace(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _in_ring(lon, lat, r):
    x, y = r[:, 0], r[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = ((y > lat) != (y2 > lat)) & (lon < (x2 - x) * (lat - y) / (y2 - y + 1e-12) + x)
    return bool(np.count_nonzero(cross) % 2)


def sea_name(lat, lon):
    """Smallest named sea / bay / gulf containing the point ('Indian Ocean' if none smaller)."""
    best, best_area = None, float("inf")
    for name, rings, area in _data()["seas"]:
        if area < best_area and any(
                r[:, 0].min() <= lon <= r[:, 0].max() and r[:, 1].min() <= lat <= r[:, 1].max()
                and _in_ring(lon, lat, r) for r in rings):
            best, best_area = name, area
    return best or "Indian Ocean"


def _km(lat0, lon0, lat1, lon1):
    p = math.pi / 180
    a = (np.sin((lat1 - lat0) * p / 2) ** 2
         + np.cos(lat0 * p) * np.cos(lat1 * p) * np.sin((lon1 - lon0) * p / 2) ** 2)
    return 12742.0 * np.arcsin(np.sqrt(a))


_DIRS8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_DIRS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(deg, points=8):
    names = _DIRS8 if points == 8 else _DIRS16
    step = 360 / len(names)
    return names[int(((deg % 360) + step / 2) // step) % len(names)]


def _bearing(lat0, lon0, lat1, lon1):
    """Compass bearing FROM point 0 TO point 1, degrees."""
    p = math.pi / 180
    dl = (lon1 - lon0) * p
    y = math.sin(dl) * math.cos(lat1 * p)
    x = math.cos(lat0 * p) * math.sin(lat1 * p) - math.sin(lat0 * p) * math.cos(lat1 * p) * math.cos(dl)
    return (math.atan2(y, x) / p) % 360


def nearest_landmarks(lat, lon, n=3):
    """Nearest distinct named places, nearest first. Each: name, country, km,
    dir (where the query point lies relative to the landmark), lat/lon of the landmark point."""
    d = _data()
    dist = _km(lat, lon, d["lat"], d["lon"])
    out, seen = [], set()
    for i in np.argsort(dist):
        nm = d["name"][i]
        if nm in seen:
            continue
        seen.add(nm)
        out.append({"name": nm, "country": d["cc"][i], "kind": d["kind"][i], "km": float(dist[i]),
                    "dir": compass(_bearing(float(d["lat"][i]), float(d["lon"][i]), lat, lon)),
                    "lat": float(d["lat"][i]), "lon": float(d["lon"][i])})
        if len(out) >= n:
            break
    return out


def describe_point(lat, lon):
    """-> {'sea', 'near' (dict), 'text'}  e.g. 'Andaman Sea · 38 km E of Little Andaman (India)'."""
    sea = sea_name(lat, lon)
    near = nearest_landmarks(lat, lon, 1)[0]
    km = near["km"]
    dist_txt = "at" if km < 1 else f"{km:.0f} km {near['dir']} of"
    country = f" ({near['country']})" if near["country"] and near["country"] != near["name"] else ""
    text = f"{sea} · {dist_txt} {near['name']}{country}".replace("at  ", "at ")
    return {"sea": sea, "near": near, "text": text}


_ANDAMAN_NICOBAR = {"Great Nicobar", "South Andaman", "Middle Andaman", "North Andaman", "Car Nicobar",
                    "Nancowry", "Katchal", "Little Nicobar", "Campbell Bay", "Little Andaman",
                    "Havelock Island", "Neil Island", "Barren Island", "Narcondam Island",
                    "Landfall Island", "Port Blair", "Diglipur", "Mayabunder"}


# ----------------------------------------------------------------------
# Conditions at scene time
# ----------------------------------------------------------------------
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"


def parse_scene_time(text):
    """'2021-06-26 12:04 UTC' -> aware datetime, or None."""
    try:
        return datetime.strptime(text.replace(" UTC", ""), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except (AttributeError, ValueError):
        return None


def _nearest_hour(when):
    h = when.replace(minute=0, second=0, microsecond=0)
    return h + timedelta(hours=1) if when.minute >= 30 else h


def _pick(block, key, idx):
    try:
        v = block[key][idx]
        return None if v is None else float(v)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def fetch_conditions(lat, lon, when_utc, timeout=8, http_get=None):
    """-> (dict of values or None, error text or None). Values may be None individually
    (e.g. no wave data over land). `http_get(url, params, timeout)` -> parsed JSON is
    injectable for tests."""
    if http_get is None:
        import requests

        def http_get(url, params, timeout):
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()

    hour = _nearest_hour(when_utc)
    key = hour.strftime("%Y-%m-%dT%H:00")
    day0 = hour.strftime("%Y-%m-%d")
    day1 = (hour + timedelta(days=1)).strftime("%Y-%m-%d")
    base = {"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": day0,
            "end_date": day1, "timezone": "GMT"}
    out, errors = {"hour_utc": key}, []

    def hourly(url, extra):
        j = http_get(url, {**base, **extra}, timeout)
        h = j["hourly"]
        return h, h["time"].index(key)

    try:
        h, i = hourly(ARCHIVE_URL, {
            "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,pressure_msl,cloud_cover,precipitation",
            "wind_speed_unit": "ms"})
        out.update(wind_ms=_pick(h, "wind_speed_10m", i), wind_dir=_pick(h, "wind_direction_10m", i),
                   gust_ms=_pick(h, "wind_gusts_10m", i), pressure_hpa=_pick(h, "pressure_msl", i),
                   cloud_pct=_pick(h, "cloud_cover", i), rain_mm=_pick(h, "precipitation", i))
    except Exception as e:                      # network, HTTP error, unexpected shape
        errors.append(f"wind/weather: {type(e).__name__}: {e}")
        if isinstance(e, OSError) and getattr(e, "response", None) is None:
            return None, errors[0]              # no connection at all: don't wait on more requests

    marine = None
    for variables in ("wave_height,wave_period,wave_direction,sea_surface_temperature",
                      "wave_height,wave_period,wave_direction"):      # retry without SST if rejected
        try:
            marine = hourly(MARINE_URL, {"hourly": variables})
            break
        except Exception as e:
            err = f"waves/sea temperature: {type(e).__name__}: {e}"
    if marine is None:
        errors.append(err)
    else:
        h, i = marine
        out.update(wave_h=_pick(h, "wave_height", i), wave_t=_pick(h, "wave_period", i),
                   wave_dir=_pick(h, "wave_direction", i), sst_c=_pick(h, "sea_surface_temperature", i))

    has_any = any(out.get(k) is not None for k in
                  ("wind_ms", "wave_h", "sst_c", "pressure_hpa", "cloud_pct"))
    if not has_any:
        return None, "; ".join(errors) or "no data returned"
    out["source"] = "Open-Meteo historical (ERA5 reanalysis)"
    return out, ("; ".join(errors) or None)


def interpret(cond, place=None):
    """Short plain-language reading of the conditions. Rules of thumb, hedged."""
    lines = []
    w = cond.get("wind_ms") if cond else None
    if w is not None:
        if w < 2:
            lines.append(f"Wind was very light ({w:.1f} m/s). A glassy sea gives radar little contrast, "
                         "so internal waves can be hard to see and some may be missed.")
        elif w <= 10:
            lines.append(f"Wind was moderate ({w:.1f} m/s), the range in which internal waves "
                         "usually show up well in radar images.")
        else:
            lines.append(f"Wind was strong ({w:.1f} m/s). Rough water can hide internal-wave "
                         "patterns, so weaker waves may be missed.")
    if place:
        near = place["near"]
        if near["name"] in _ANDAMAN_NICOBAR and near["km"] <= 250:
            lines.append(f"This is {near['km']:.0f} km {near['dir']} of {near['name']}, on the "
                         "Andaman-Nicobar island chain. Tides flowing over this underwater ridge are a "
                         "known source of strong internal waves in the Andaman Sea.")
        else:
            lines.append("Internal waves in this region are usually generated where tides flow over "
                         "the continental shelf edge or underwater ridges.")
    return lines


# ----------------------------------------------------------------------
def selftest():
    p = describe_point(8.0, 94.6)                      # east of Nicobar Is. (Andaman Sea side)
    assert p["sea"] in ("Andaman Sea", "Bay of Bengal"), p
    assert p["near"]["name"] in _ANDAMAN_NICOBAR | {"Nancowry", "Katchal"}, p
    p2 = describe_point(15.0, 87.0)
    assert p2["sea"] == "Bay of Bengal", p2
    assert describe_point(10.5, 66.0)["sea"] in ("Arabian Sea", "Indian Ocean")
    assert compass(0) == "N" and compass(225) == "SW" and compass(359) == "N"
    t = parse_scene_time("2021-06-26 12:04 UTC")
    assert _nearest_hour(t).hour == 12 and _nearest_hour(t.replace(minute=40)).hour == 13
    assert parse_scene_time(None) is None

    def fake_get(url, params, timeout):
        times = [f"2021-06-26T{h:02d}:00" for h in range(24)] + [f"2021-06-27T{h:02d}:00" for h in range(24)]
        n = len(times)
        if "archive" in url:
            return {"hourly": {"time": times, "wind_speed_10m": [4.0] * n, "wind_direction_10m": [225.0] * n,
                               "wind_gusts_10m": [6.0] * n, "pressure_msl": [1008.0] * n,
                               "cloud_cover": [60] * n, "precipitation": [0.2] * n}}
        if "sea_surface_temperature" in params["hourly"]:
            raise RuntimeError("400 unknown variable")      # exercises the retry
        return {"hourly": {"time": times, "wave_height": [1.4] * n, "wave_period": [7.0] * n,
                           "wave_direction": [200.0] * n}}
    c, err = fetch_conditions(8.0, 94.6, t, http_get=fake_get)
    assert c["wind_ms"] == 4.0 and c["wave_h"] == 1.4 and c["sst_c"] is None, c
    assert c["hour_utc"] == "2021-06-26T12:00"
    c2, err2 = fetch_conditions(8.0, 94.6, t, http_get=lambda *a: (_ for _ in ()).throw(OSError("offline")))
    assert c2 is None and "offline" in err2
    assert len(interpret(c, p)) == 2
    print("oceanxray_context self-test: all checks passed")
    print(p["text"]); print(p2["text"])




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        print(describe_point(float(sys.argv[1]), float(sys.argv[2]))["text"] if len(sys.argv) == 3
              else "usage: python oceanxray_context.py selftest | <lat> <lon>")