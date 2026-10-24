"""
OceanXRay -- Internal Wave Detection Dashboard

Upload a Sentinel-1 tile, run the improved detector (Faster R-CNN v2,
92.3% mAP@0.5 on the S1-IW-2023 test split), see captioned boxes, and --
for tiles from our own pipeline -- watch the map fly from the study region
to where the tile actually is.

Run with: streamlit run iw_dashboard.py   (data files go in ./data, see README)

Files expected next to this script
----------------------------------
required  ckpt_improved_v2_s2.pth                     the detector
optional  oceanxray_geo.py + scene_geo_table.json     geolocating uploads
optional  land_ocean_mask.npz                         'near land' warnings
optional  oceanxray_context.py                        place names + weather at scene time
optional  real_internal_wave_detections_v2.parquet    detection spots on the maps
          (falls back to real_internal_wave_detections.parquet, old model)
optional  sar_era5_sst_spatial_matches_june.parquet   SST/wind map tab

Speed notes
-----------
Streamlit re-runs this whole script on every click, including the code of
tabs you are not looking at. So: the model is loaded once (cache_resource),
detector output is cached per uploaded file (moving the threshold slider
only re-filters boxes, it never re-runs the model), and the cartopy map is
drawn once and reused as a PNG.
"""

import hashlib
import io
from html import escape as html_escape
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

try:
    import oceanxray_geo as geo
except ImportError:            # dashboard still works, just without geolocation
    geo = None

try:
    import oceanxray_context as ctx   # place names + scene-time weather
except ImportError:                   # dashboard still works, with coordinates only
    ctx = None

try:
    import oceanxray_bathy as bathy   # GEBCO sea-floor depth (embedded grid)
except ImportError:                   # dashboard still works without the depth panel
    bathy = None

st.set_page_config(page_title="OceanXRay -- Internal Wave Detector", layout="wide")

# Data files live in ./data next to this script (override with the OCEANXRAY_DATA env var).
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("OCEANXRAY_DATA", os.path.join(APP_DIR, "data"))
CHECKPOINT_PATH = os.path.join(DATA_DIR, "ckpt_improved_v2_s2.pth")
GEO_TABLE_PATH = os.path.join(DATA_DIR, "scene_geo_table.json")
LAND_MASK_PATH = os.path.join(DATA_DIR, "land_ocean_mask.npz")
SPATIAL_MATCHES_PATH = os.path.join(DATA_DIR, "sar_era5_sst_spatial_matches_june.parquet")
DETECTIONS_V2_PATH = os.path.join(DATA_DIR, "real_internal_wave_detections_v2.parquet")
DETECTIONS_OLD_PATH = os.path.join(DATA_DIR, "real_internal_wave_detections.parquet")
BOX_COLOR = (255, 196, 0)          # amber: readable on grey SAR
NEAR_LAND_COLOR = (156, 163, 175)  # grey: box within 1 km of land
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI = (69.0, 4.0, 96.0, 24.0)      # lon_min, lat_min, lon_max, lat_max


# ----------------------------------------------------------------------
# Cached resources (built once per server process)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the detector...")
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, min_size=1024, max_size=1024)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


@st.cache_resource
def load_geo():
    """-> (transforms, raw table, is_on_land or None), or None if unavailable."""
    if geo is None or not os.path.exists(GEO_TABLE_PATH):
        return None
    transforms, raw = geo.load_geo_table(GEO_TABLE_PATH)
    land = geo.load_land_checker(LAND_MASK_PATH) if os.path.exists(LAND_MASK_PATH) else None
    return transforms, raw, land


@st.cache_data(show_spinner="Running the detector...", max_entries=32)
def detect(image_bytes: bytes):
    """All boxes + scores for one image. Cached by file content, so changing
    the threshold (or any other widget) never re-runs the model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        out = load_model()(TF.to_tensor(image).unsqueeze(0).to(DEVICE))[0]
    return out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), image.size


def files_stamp(*paths):
    """Modification times, passed to cached functions so they refresh when a file changes."""
    return tuple(os.path.getmtime(p) if os.path.exists(p) else None for p in paths)


@st.cache_data
def load_detections(stamp=None):
    """Detection spots for the maps -> (DataFrame[lat, lon, confidence, scene_id], note)."""
    if os.path.exists(DETECTIONS_V2_PATH):
        df = pd.read_parquet(DETECTIONS_V2_PATH)
        df = df[df["kept"]] if "kept" in df else df
        note = "improved model, open water only (>1 km from land)"
    elif os.path.exists(DETECTIONS_OLD_PATH):
        df = pd.read_parquet(DETECTIONS_OLD_PATH)
        note = ("OLD model output, includes coastal false positives -- "
                "re-run infer_iw_on_real_scenes.py to refresh")
    else:
        return pd.DataFrame(columns=["real_lat", "real_lon", "confidence", "scene_id"]), "none found"
    df = df.dropna(subset=["real_lat", "real_lon"])
    return df[["real_lat", "real_lon", "confidence", "scene_id"]].reset_index(drop=True), note


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def _font(size):
    for f in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def trim_nodata(img, dets):
    """Crop away all-black rows/columns (outside the radar swath) for display only.

    Returns the cropped image and the detections with boxes shifted to match. The
    detection itself always runs on the full, unmodified tile."""
    a = np.asarray(img.convert("L"))
    valid = a > 3
    rows, cols = np.where(valid.any(axis=1))[0], np.where(valid.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img, dets
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    for d in dets:                                          # never crop away a detection box
        bx1, by1, bx2, by2 = d["box"]
        x0, y0 = min(x0, int(bx1)), min(y0, int(by1))
        x1, y1 = max(x1, int(np.ceil(bx2))), max(y1, int(np.ceil(by2)))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, a.shape[1]), min(y1, a.shape[0])
    if (y1 - y0) * (x1 - x0) > 0.95 * a.shape[0] * a.shape[1]:
        return img, dets                                   # nothing worth trimming
    shifted = []
    for d in dets:
        bx1, by1, bx2, by2 = d["box"]
        shifted.append(dict(d, box=[bx1 - x0, by1 - y0, bx2 - x0, by2 - y0]))
    return img.crop((x0, y0, x1, y1)), shifted


def draw_detections(img, dets):
    """Box + filled caption tag 'Internal wave 87%' (grey + 'near land' if flagged)."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    W = img.width
    font = _font(max(14, W // 45))
    lw = max(3, W // 300)
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        color = NEAR_LAND_COLOR if d["near_land"] else BOX_COLOR
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = f"Internal wave {d['score'] * 100:.0f}%" + (" · near land" if d["near_land"] else "")
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), label, font=font)
        tw, th = tx1 - tx0, ty1 - ty0
        pad = max(3, th // 4)
        ty = y1 - th - 2 * pad if y1 - th - 2 * pad >= 0 else y1
        tx = min(x1, W - tw - 2 * pad)
        draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=color)
        draw.text((tx + pad, ty + pad - ty0), label, fill="white", font=font)
    return img


def place_for(lon, lat):
    """{'sea', 'near', 'text'} for a point, or None if the context module is missing."""
    if ctx is None:
        return None
    try:
        return ctx.describe_point(lat, lon)
    except Exception:
        return None


def view_box(tile_fp, landmarks, min_deg=0.9):
    """Map view that holds the tile and its nearest landmark(s): [[south, west], [north, east]].
    At least ~100 km across so the tile is never filling the screen."""
    lons = [p[0] for p in tile_fp] + [m["lon"] for m in landmarks]
    lats = [p[1] for p in tile_fp] + [m["lat"] for m in landmarks]
    w, e, s, n = min(lons), max(lons), min(lats), max(lats)
    if e - w < min_deg:
        c = (e + w) / 2
        w, e = c - min_deg / 2, c + min_deg / 2
    if n - s < min_deg:
        c = (n + s) / 2
        s, n = c - min_deg / 2, c + min_deg / 2
    return [[s, w], [n, e]]


def geolocate(filename, image_size, boxes, scores, threshold):
    """-> (dets, tile_info or None, reason-if-not-geolocated)."""
    keep = scores >= threshold
    dets = [{"box": [float(v) for v in b], "score": float(s), "near_land": False, "poly": None}
            for b, s in zip(boxes[keep], scores[keep])]
    G = load_geo()
    parsed = geo.parse_tile_filename(filename) if geo is not None else None
    if parsed is None:
        return dets, None, ("This file name isn't one of our pipeline tiles "
                            "(`<scene>_tileNNNN_rowR_colC.png`), so its location is unknown.")
    if G is None:
        return dets, None, (f"Geolocation needs `oceanxray_geo.py` and `{GEO_TABLE_PATH}` "
                            "next to the dashboard.")
    transforms, raw, is_on_land = G
    scene_id, tile_idx, row0, col0 = parsed
    T = transforms.get(scene_id)
    if T is None:
        return dets, None, f"Scene `{scene_id[:40]}…` is not in `{GEO_TABLE_PATH}`."

    # Boxes come back in the uploaded image's pixels; the geometry assumes
    # the pipeline's 1024 px tiles, so rescale if the upload was resized.
    sx = geo.OUTPUT_PIXELS / image_size[0]
    sy = geo.OUTPUT_PIXELS / image_size[1]
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        b = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        d["poly"] = geo.box_polygon(T, row0, col0, b)
        d["center"] = geo.box_center(T, row0, col0, b)
        if is_on_land is not None:
            d["near_land"] = bool(geo.box_touches_land(T, row0, col0, b, is_on_land))
        d["place"] = place_for(*d["center"])
    center = geo.box_center(T, row0, col0, [0, 0, geo.OUTPUT_PIXELS, geo.OUTPUT_PIXELS])
    tile_fp = geo.tile_footprint(T, row0, col0)
    landmarks = []
    if ctx is not None:
        try:
            landmarks = [m for m in ctx.nearest_landmarks(center[1], center[0], 3) if m["km"] <= 220]
        except Exception:
            landmarks = []
    info = {
        "scene_fp": T.footprint(),
        "tile_fp": tile_fp,
        "center": center,
        "time": raw[scene_id].get("time") or geo.scene_datetime_str(scene_id),
        "tile_idx": tile_idx,
        "place": place_for(*center),
        "landmarks": landmarks,
        "view": view_box(tile_fp, landmarks[:2]),
    }
    return dets, info, None


def fmt_latlon(lon, lat):
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"


# ----------------------------------------------------------------------
# Animated map (Leaflet in an iframe)
# ----------------------------------------------------------------------
MAP_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body,#map{margin:0;height:100%;background:#0b2239;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 .card{background:rgba(11,26,43,.88);color:#e6edf5;padding:10px 12px;border-radius:8px;font-size:13px;line-height:1.45;max-width:270px;box-shadow:0 1px 4px rgba(0,0,0,.4)}
 .card b{color:#fff}.card .q{color:#9fb3c8;font-size:12px}
 .legend div{display:flex;align-items:center;gap:6px;margin:0}
 .card.legend{display:flex;flex-wrap:wrap;align-items:center;gap:4px 16px;max-width:calc(100vw - 20px);box-sizing:border-box;padding:6px 10px;font-size:12px;line-height:1.3}
 .card.legend .q{font-size:11px}
 .leaflet-bottom.leaflet-left .leaflet-control{margin-bottom:22px}   /* sits just above the attribution line */
 .sw{width:14px;height:10px;border-radius:2px;display:inline-block}
 .mini{float:right;background:none;border:0;color:#9fb3c8;cursor:pointer;font-size:16px;line-height:1;padding:0 0 0 10px}
 .mini:hover{color:#fff}
 .card.min .more{display:none}
 .btn{background:#ffc400;color:#111;border:0;border-radius:6px;padding:6px 10px;font-weight:600;cursor:pointer;font-size:12px}
 .leaflet-tooltip.iw{background:#ffc400;color:#111;border:0;font-weight:600;font-size:12px;padding:2px 6px;box-shadow:none}
 .leaflet-tooltip.iw.land{background:#9ca3af}
 .leaflet-tooltip.iw:before{display:none}
 .leaflet-tooltip.lm{background:rgba(11,26,43,.88);color:#fff;border:0;font-size:11px;padding:2px 6px;box-shadow:none}
 .leaflet-tooltip.lm:before{display:none}
</style></head><body><div id="map"></div><script>
const P = __PAYLOAD__;
const ll = pts => pts.map(p => [p[1], p[0]]);
const map = L.map('map', {zoomSnap: 0.25, worldCopyJump: false, preferCanvas: true, maxZoom: 13});

// Base maps. maxNativeZoom = the deepest level the provider really has tiles for in open
// water; beyond it Leaflet enlarges the last good tile instead of showing "data not available".
const esri = 'https://server.arcgisonline.com/ArcGIS/rest/services/';
const tl = (u, nat, attr) => L.tileLayer(u, {maxNativeZoom: nat, maxZoom: 13, attribution: attr || ''});
const BASE = {
  'Ocean (Esri)': L.layerGroup([
    tl(esri + 'Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', 10, 'Tiles &copy; Esri, GEBCO, NOAA, Garmin, HERE'),
    tl(esri + 'Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}', 10)]),
  'Satellite (Esri)': L.layerGroup([
    tl(esri + 'World_Imagery/MapServer/tile/{z}/{y}/{x}', 12, 'Imagery &copy; Esri, Maxar, Earthstar Geographics'),
    tl(esri + 'Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', 12)]),
  'Light map (CARTO)': L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
    {subdomains: 'abcd', maxNativeZoom: 14, maxZoom: 13,
     attribution: '&copy; OpenStreetMap contributors &copy; CARTO'})
};
BASE['Ocean (Esri)'].addTo(map);
L.control.layers(BASE, null, {position: 'topleft'}).addTo(map);

const roi = [[P.roi[1], P.roi[0]], [P.roi[3], P.roi[2]]];
L.rectangle(roi, {color: '#ffffff', weight: 1.2, dashArray: '6 6', fill: false, interactive: false}).addTo(map);

// Other detections: only drawn when zoomed out, never on top of this tile's own boxes.
const spots = L.layerGroup();
const tileBox = P.tile ? L.latLngBounds(ll(P.tile.tile_fp)) : null;
P.spots.forEach(s => {
  if (tileBox && tileBox.contains([s[0], s[1]])) return;
  L.circleMarker([s[0], s[1]], {radius: 3.5, color: '#0e7490', weight: 1, fillColor: '#22d3ee', fillOpacity: .85})
   .bindTooltip('Internal wave ' + Math.round(s[2] * 100) + '%').addTo(spots);
});
let spotMaxZoom = null;                       // null = always visible (no tile to focus on)
const viewB = P.tile ? L.latLngBounds(P.tile.view) : null;
const PAD = [40, 40], ZMAX = 10;
// keep the tile and its boxes in the part of the map not covered by the info card (top right)
// and the legend (bottom left); on a small map fall back to even padding
function fitPad() {
  const sz = map.getSize(), c = document.getElementById('card'), g = document.getElementById('legend');
  const ch = c ? c.offsetHeight : 40, lh = g ? g.offsetHeight : 0;
  if (sz.x < 300 || sz.y - ch - lh < 200) return {padding: PAD};
  return {paddingTopLeft: [60, ch + 24], paddingBottomRight: [24, lh + 24]};
}
function padPoint() { const f = fitPad(); return f.padding ? L.point(2 * PAD[0], 2 * PAD[1])
  : L.point(f.paddingTopLeft[0] + f.paddingBottomRight[0], f.paddingTopLeft[1] + f.paddingBottomRight[1]); }
function armSpots() { if (viewB) spotMaxZoom = Math.min(map.getBoundsZoom(viewB, false, padPoint()), ZMAX) - 0.75; }
function updateSpots() {
  const show = spotMaxZoom === null || map.getZoom() <= spotMaxZoom;
  if (show) { if (!map.hasLayer(spots)) spots.addTo(map); }
  else if (map.hasLayer(spots)) map.removeLayer(spots);
}
map.on('zoom zoomend', updateSpots);
let userMoved = false, settled = false;
map.on('resize', () => { if (viewB && settled && !userMoved) {
  map.fitBounds(viewB, Object.assign({maxZoom: ZMAX, animate: false}, fitPad())); armSpots(); updateSpots(); } });
['mousedown', 'wheel', 'touchstart', 'keydown'].forEach(ev =>
  map.getContainer().addEventListener(ev, () => { userMoved = true; }, {passive: true}));

// legend
const legend = L.control({position: 'bottomleft'});
legend.onAdd = () => { const d = L.DomUtil.create('div', 'card legend'); d.id = 'legend'; d.innerHTML =
  '<div><span class="sw" style="border:1.5px dashed #fff"></span>Study region</div>' +
  '<div><span class="sw" style="background:#22d3ee;border-radius:50%;width:10px"></span>' +
    (P.tile ? 'Our other detections (' + P.spots.length + '): zoom out to see' : 'Detections in our scenes (' + P.spots.length + ')') + '</div>' +
  (P.tile ? '<div><span class="sw" style="border:2px solid #fff"></span>This tile (25 km)</div>' +
  '<div><span class="sw" style="background:#ffc400"></span>Detected internal wave</div>' +
  (P.tile.boxes.some(b => b.near_land) ? '<div><span class="sw" style="background:#9ca3af"></span>Near land (likely false)</div>' : '') : '') +
  '<div class="q">' + P.spots_note + '</div>'; return d; };
legend.addTo(map);

// info card
const card = L.control({position: 'topright'});
card.onAdd = () => { const d = L.DomUtil.create('div', 'card'); d.id = 'card'; L.DomEvent.disableClickPropagation(d); return d; };
card.addTo(map);
const cardEl = document.getElementById('card');

const overlay = L.layerGroup().addTo(map);
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// run fn once the map stops moving (with a fallback in case no event fires)
function after(fn, fallbackMs) { let done = false; const go = () => { if (!done) { done = true; fn(); } };
  map.once('moveend', go); setTimeout(go, fallbackMs); }

function drawScene() { if (P.tile.scene_fp) L.polygon(ll(P.tile.scene_fp), {color: '#cbd5e1', weight: 1,
  dashArray: '3 5', fill: false, interactive: false}).addTo(overlay); }
function drawTile() { L.polygon(ll(P.tile.tile_fp), {color: '#ffffff', weight: 2, fill: false, interactive: false}).addTo(overlay); }
function drawLandmarks() {
  (P.tile.landmarks || []).forEach((m, i) => {
    if (i === 0) L.polyline([P.tile.center_ll, [m.lat, m.lon]], {color: '#ffffff', weight: 1.2, dashArray: '2 6',
      opacity: .8, interactive: false}).addTo(overlay);
    L.circleMarker([m.lat, m.lon], {radius: i ? 4 : 5, color: '#0b1a2b', weight: 1.5, fillColor: '#ffffff', fillOpacity: 1})
     .bindTooltip(m.name + ' · ' + m.km + ' km', {permanent: true, direction: i % 2 ? 'left' : 'right',
                  offset: [i % 2 ? -6 : 6, 0], className: 'lm'})
     .addTo(overlay);
  });
}
function drawBoxes(stagger) {
  const many = P.tile.boxes.length > 6;
  P.tile.boxes.forEach((b, i) => setTimeout(() => {
    const c = b.near_land ? '#9ca3af' : '#ffc400';
    const lbl = 'Internal wave ' + Math.round(b.score * 100) + '%' + (b.near_land ? ' · near land' : '');
    L.polygon(ll(b.poly), {color: c, weight: 2.5, fillColor: c, fillOpacity: .18, dashArray: b.near_land ? '5 4' : null})
     .bindTooltip(lbl, {permanent: !many, direction: 'top', className: 'iw' + (b.near_land ? ' land' : '')})
     .addTo(overlay);
  }, stagger ? i * 260 : 0));
}
function showCard() {
  const t = P.tile, n = t.boxes.length;
  settled = true;
  cardEl.classList.add('min');          // collapsed by default; "+" opens it for a demo
  cardEl.innerHTML = '<button class="mini" id="mini" title="Expand">+</button>' +
    '<b>' + (t.time || 'Sentinel-1 scene') + '</b><div class="more">' +
    (t.place_text ? '<span id="pl"></span><br>' : '') +
    'Tile centre ' + t.center_txt + '<br>' +
    (n ? n + ' internal-wave box' + (n > 1 ? 'es' : '') + ' at &ge; ' + P.threshold_pct + '%'
       : 'No boxes at &ge; ' + P.threshold_pct + '%') +
    '<div class="q" style="margin-top:6px">Position from the scene\'s ground-control points; approximate.</div>' +
    '<div style="margin-top:8px"><button class="btn" id="replay">&#8635; Replay</button></div></div>';
  if (t.place_text) document.getElementById('pl').textContent = t.place_text;    // text, never HTML
  document.getElementById('replay').onclick = play;
  const mini = document.getElementById('mini');
  mini.onclick = () => { const m = cardEl.classList.toggle('min');
    mini.innerHTML = m ? '+' : '&minus;'; mini.title = m ? 'Expand' : 'Minimise'; };
}

function play() {
  settled = false; userMoved = false;
  overlay.clearLayers(); cardEl.classList.remove('min'); cardEl.innerHTML = '<span class="q">Locating tile&hellip;</span>';
  map.fitBounds(roi, {animate: false, padding: [10, 10]});
  armSpots(); updateSpots();
  setTimeout(() => {
    const sceneB = P.tile.scene_fp ? L.latLngBounds(ll(P.tile.scene_fp)) : null;
    const toTile = () => { map.flyToBounds(viewB, Object.assign({duration: 1.8, maxZoom: ZMAX}, fitPad()));
      after(() => { drawTile(); drawLandmarks(); drawBoxes(true); showCard(); }, 2600); };
    if (sceneB) { map.flyToBounds(sceneB, {duration: 1.6, padding: [30, 30]});
      after(() => { drawScene(); setTimeout(toTile, 350); }, 2300); }
    else toTile();
  }, 650);
}

if (!P.tile) {
  map.fitBounds(roi, {padding: [10, 10]});
  updateSpots();
  cardEl.innerHTML = '<b>Location unknown</b><br><span class="q"></span>';
  cardEl.querySelector('.q').textContent = P.message.replace(/`/g, '');   // text, never HTML
} else if (P.animate && !reduce) {
  play();
} else {
  map.fitBounds(viewB, Object.assign({maxZoom: ZMAX, animate: false}, fitPad()));
  armSpots(); updateSpots();
  drawScene(); drawTile(); drawLandmarks(); drawBoxes(false); showCard();
}
</script></body></html>"""


def render_geomap(dets, tile_info, message, threshold, animate, height=600):
    spots, note = load_detections(files_stamp(DETECTIONS_V2_PATH, DETECTIONS_OLD_PATH))
    payload = {
        "roi": ROI,
        "spots": spots[["real_lat", "real_lon", "confidence"]].round(5).values.tolist(),
        "spots_note": f"Detection spots: {note}",
        "threshold_pct": int(round(threshold * 100)),
        "animate": bool(animate),
        "message": message or "",
        "tile": None,
    }
    if tile_info is not None:
        payload["tile"] = {
            "scene_fp": tile_info["scene_fp"], "tile_fp": tile_info["tile_fp"],
            "time": tile_info["time"],
            "center_txt": fmt_latlon(*tile_info["center"]),
            "center_ll": [tile_info["center"][1], tile_info["center"][0]],
            "place_text": tile_info["place"]["text"] if tile_info.get("place") else "",
            "view": tile_info["view"],
            "landmarks": [{"name": m["name"], "km": int(round(m["km"])), "lat": m["lat"], "lon": m["lon"]}
                          for m in tile_info.get("landmarks", [])],
            "boxes": [{"poly": d["poly"], "score": d["score"], "near_land": d["near_land"]}
                      for d in dets],
        }
    html = MAP_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/"))
    if hasattr(st, "iframe"):          # newer Streamlit (components.html is being retired)
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


# ----------------------------------------------------------------------
# Static SST / wind map (tab 2), drawn once
# ----------------------------------------------------------------------
LANDMARKS = [
    ("Chennai", 80.27, 13.08, False), ("Kolkata", 88.36, 22.57, False),
    ("Visakhapatnam", 83.30, 17.69, False), ("Colombo", 79.86, 6.93, False),
    ("Port Blair", 92.73, 11.62, False), ("Andaman-Nicobar ridge", 93.0, 10.0, True),
    ("Bay of Bengal", 87.0, 15.0, True), ("Andaman Sea", 95.5, 12.0, True),
]


@st.cache_data(show_spinner="Drawing the map (first time only)...")
def multimodal_map_png(stamp=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_parquet(SPATIAL_MATCHES_PATH).dropna(
        subset=["real_lon", "real_lat", "sst_sst", "era5_u10", "era5_v10"])
    det, note = load_detections(files_stamp(DETECTIONS_V2_PATH, DETECTIONS_OLD_PATH))
    sst_c = df["sst_sst"] - 273.15
    sub = df.iloc[::max(1, len(df) // 350)]

    def draw(use_cartopy):
        if use_cartopy:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            fig = plt.figure(figsize=(11, 9))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([ROI[0], ROI[2], ROI[1], ROI[3]], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.OCEAN, facecolor="#dbeafe")
            ax.add_feature(cfeature.LAND, facecolor="#e8e0d0")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#555")
            kw = {"transform": ccrs.PlateCarree()}
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
            gl.top_labels = gl.right_labels = False
        else:
            fig, ax = plt.subplots(figsize=(11, 9))
            ax.set_facecolor("#dbeafe")
            ax.set_xlim(ROI[0], ROI[2]); ax.set_ylim(ROI[1], ROI[3]); ax.set_aspect("equal")
            kw = {}

        sc = ax.scatter(df["real_lon"], df["real_lat"], c=sst_c, cmap="RdYlBu_r", s=8, alpha=0.6,
                        edgecolors="none", vmin=26, vmax=31, zorder=3, **kw)
        ax.quiver(sub["real_lon"].values, sub["real_lat"].values, sub["era5_u10"].values,
                  sub["era5_v10"].values, color="black", alpha=0.35, scale=160, width=0.0018,
                  zorder=4, **kw)
        ax.scatter(det["real_lon"], det["real_lat"], s=20, c="lime", edgecolors="darkgreen",
                   linewidths=0.4, alpha=0.85, label=f"Internal wave detections ({len(det)})",
                   zorder=5, **kw)
        for name, lon, lat, water in LANDMARKS:
            color = "navy" if water else "black"
            ax.plot(lon, lat, marker="*" if water else "^", color=color, markersize=11, zorder=6, **kw)
            ax.text(lon + 0.2, lat + 0.2, name, fontsize=9, color=color, zorder=6,
                    style="italic" if water else "normal", **kw)
        fig.colorbar(sc, ax=ax, label="Sea surface temperature (°C)", shrink=0.7)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_title("June 2021: SST (colour) + ERA5 wind (arrows) + internal-wave detections (green)")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    try:
        return draw(True), note
    except Exception:
        # cartopy missing, or it could not download the coastlines (offline): plain map instead
        plt.close("all")
        return draw(False), note


# ----------------------------------------------------------------------
# Conditions at scene time (live weather, with an offline fallback)
# ----------------------------------------------------------------------
@st.cache_resource
def _conditions_store():
    return {}


def get_conditions(lat, lon, time_text):
    """-> (values or None, error text or None). Successes are kept for the life of the
    server; failures for 2 minutes, so being offline doesn't slow every click."""
    import time as _time
    when = ctx.parse_scene_time(time_text)
    if when is None:
        return None, "scene time unknown"
    key = (round(lat * 4) / 4, round(lon * 4) / 4, ctx._nearest_hour(when).isoformat())
    store = _conditions_store()
    hit = store.get(key)
    if hit and (hit[1] is not None or _time.time() - hit[0] < 120):
        return hit[1], hit[2]
    cond, err = ctx.fetch_conditions(lat, lon, when)
    store[key] = (_time.time(), cond, err)
    return cond, err


@st.cache_data
def load_matches(stamp=None):
    cols = ["real_lon", "real_lat", "era5_u10", "era5_v10", "sst_sst"]
    return pd.read_parquet(SPATIAL_MATCHES_PATH, columns=cols).dropna().to_numpy(dtype=float)


def monthly_nearby(lat, lon, radius_deg=0.5):
    """Offline fallback: June 2021 monthly ERA5 wind and SST from our own matched patches
    within ~55 km. -> values or None."""
    if not os.path.exists(SPATIAL_MATCHES_PATH):
        return None
    try:
        a = load_matches(files_stamp(SPATIAL_MATCHES_PATH))
    except Exception:
        return None
    near = a[(a[:, 0] - lon) ** 2 + (a[:, 1] - lat) ** 2 <= radius_deg ** 2]
    if len(near) == 0:
        return None
    u, v = float(near[:, 2].mean()), float(near[:, 3].mean())
    return {"wind_ms": float(np.hypot(u, v)),
            "wind_dir": float((270 - np.degrees(np.arctan2(v, u))) % 360),
            "sst_c": float(near[:, 4].mean() - 273.15), "n": int(len(near))}


def short_reason(err):
    e = (err or "").lower()
    if any(k in e for k in ("proxy", "connection", "timeout", "resolve", "network", "unreachable")):
        return "no connection to the weather service"
    return "the weather service returned an error"


def show_seafloor(tile_info):
    """What the sea floor looks like under this tile (GEBCO), in plain language."""
    if tile_info is None or bathy is None:
        return
    lon, lat = tile_info["center"]
    sf = bathy.describe_seafloor(lat, lon)
    if sf is None or sf["depth_m"] <= 0:
        return
    st.subheader("Sea floor under this tile")
    c1, c2, c3 = st.columns(3)
    c1.metric("Water depth", f"{sf['depth_m']:,.0f} m")
    c2.metric("Zone", sf["zone"].capitalize())
    sb = sf["shelf_break_km"]
    c3.metric("Distance to shelf edge (200 m)", "n/a" if sb is None else f"{sb:.0f} km")
    notes = []
    if sf["slope_m_per_km"] >= 40:
        notes.append("The sea floor is steep nearby. Tides pushed over steep slopes, ridges and shelf "
                     "edges are a common place for internal waves to form, so this fits what the detector found.")
    elif sf["zone"] in ("very shallow shelf", "continental shelf"):
        notes.append("Shallow, gently sloping shelf. Internal waves here usually arrived from deeper water "
                     "or formed at the shelf edge and travelled in.")
    else:
        notes.append("Open deep water. Internal waves here usually travel a long way from where they formed.")
    for n in notes:
        st.markdown(f"- {n}")
    st.caption("Depth from the GEBCO global grid, resampled to about 5.5 km, so narrow canyons and "
               "small features are smoothed out. GEBCO Compilation Group.")


def show_conditions(tile_info):
    """Wind, waves and sea temperature for this tile's date, plus a plain-language reading."""
    if tile_info is None or ctx is None or not tile_info.get("time"):
        return
    st.subheader("Conditions when this was captured")
    lon, lat = tile_info["center"]
    with st.spinner("Looking up the weather for this date and place..."):
        cond, err = get_conditions(lat, lon, tile_info["time"])
    monthly = None
    if cond is None:
        monthly = monthly_nearby(lat, lon)
        shown = monthly
    else:
        shown = cond
    if shown is None:
        st.info(f"Weather for this date isn't available right now: {short_reason(err)}.")
        if err:
            with st.expander("Technical detail"):
                st.code(err)
        return

    def num(v, fmt, unit=""):
        return "n/a" if v is None else f"{v:{fmt}}{unit}"
    wind_dir = shown.get("wind_dir")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wind" + (f" (from {ctx.compass(wind_dir)})" if wind_dir is not None else ""),
              num(shown.get("wind_ms"), ".1f", " m/s"))
    if monthly is None:
        c2.metric("Waves (height · period)",
                  "n/a" if shown.get("wave_h") is None else
                  f"{shown['wave_h']:.1f} m · {num(shown.get('wave_t'), '.0f', ' s')}")
        c4.metric("Cloud · rain", f"{num(shown.get('cloud_pct'), '.0f', '%')} · {num(shown.get('rain_mm'), '.1f', ' mm/h')}")
    else:
        c2.metric("Waves (height · period)", "n/a")
        c4.metric("Cloud · rain", "n/a")
    c3.metric("Sea-surface temperature", num(shown.get("sst_c"), ".1f", " °C"))

    if monthly is None:
        st.caption(f"{cond['source']} · nearest hour {cond['hour_utc'].replace('T', ' ')} UTC · about 25 km grid cell at "
                   "the tile centre. Reanalysis values (a model constrained by observations), not a "
                   "measurement at the exact spot. Weather data by Open-Meteo.com (CC BY 4.0).")
        if err:                      # e.g. waves or sea temperature were unavailable
            with st.expander("Some values were unavailable"):
                st.code(err)
    else:
        st.caption(f"Live weather unavailable ({short_reason(err)}). Showing the June 2021 monthly average "
                   f"from our own ERA5 / SST data within ~55 km ({monthly['n']} patches), not this exact day.")
        if err:
            with st.expander("Technical detail"):
                st.code(err)
    for line in ctx.interpret(shown if monthly is None else {}, tile_info.get("place")):
        st.markdown(f"- {line}")


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.markdown("""<style>
/* tighter page: less empty space above the title and between blocks */
.block-container {padding-top: 1.2rem; padding-bottom: 1.5rem;}
div[data-testid="stVerticalBlock"] {gap: 0.6rem;}
/* centred heading and one-line descriptions */
.ox-title {text-align: center; margin: 0 0 0.2rem 0; font-size: 1.85rem; font-weight: 700; line-height: 1.2;}
.ox-desc {text-align: center; color: rgba(128,128,128,0.95); font-size: 0.9rem; margin: 0 0 0.4rem 0;}
/* detection image shrinks with the screen height, so the table and map stay in view */
.st-key-ox_left div[data-testid="stImage"] img {max-height: max(220px, calc(100vh - 560px)); width: auto !important;
                                                 max-width: 100%; object-fit: contain; margin: 0 auto; display: block;}
.ox-hint {font-size: 0.78rem; line-height: 1.3; color: rgba(128,128,128,0.95); margin: -0.5rem 0 0.4rem 0;}
/* live map fills the rest of the screen height (Leaflet re-fits on resize) */
.st-key-ox_right iframe {height: max(380px, calc(100vh - 290px)) !important;}
.st-key-ox_right [data-testid="stElementContainer"]:has(> iframe) {height: auto !important;}
.st-key-ox_right [data-testid="stCaptionContainer"] {margin-top: -0.5rem;}
</style>""", unsafe_allow_html=True)


def centered_title(text):
    st.markdown(f'<h1 class="ox-title">{html_escape(text)}</h1>', unsafe_allow_html=True)


def centered_desc(text):
    st.markdown(f'<p class="ox-desc">{html_escape(text)}</p>', unsafe_allow_html=True)


centered_title("OceanXRay -- Multimodal Ocean Dashboard")

if not os.path.exists(CHECKPOINT_PATH):
    st.error(f"Checkpoint not found at '{CHECKPOINT_PATH}'. Put ckpt_improved_v2_s2.pth "
             "in the data/ folder next to this script (see README).")
    st.stop()
load_model()

UPLOAD_HELP = ("Upload a Sentinel-1 tile (from convert_scenes_to_images.py, keep the original "
               "file name to geolocate it) or an S1-IW-2023 test image")
MAP_HEIGHT = 600
threshold = 0.70          # fixed confidence threshold (no slider)

tab1, tab2 = st.tabs(["Internal wave detector (live)", "Integrated multimodal map"])

with tab1:
    centered_desc("Faster R-CNN v2 (improved training), fine-tuned on S1-IW-2023: "
                  "92.3% mAP@0.5 on the test split (mean of 3 runs).")
    # Stacked layout: upload, then detection, on the left; the live map uses the right half.
    left, right = st.columns([5, 6], gap="medium")
    try:                                   # keyed containers let the CSS above target each side
        ox_left = left.container(key="ox_left")
        ox_right = right.container(key="ox_right")
    except TypeError:                      # older Streamlit: same layout, default sizes
        ox_left, ox_right = left.container(), right.container()
    upload_key = f"tile_upload_{st.session_state.get('upload_round', 0)}"
    with ox_left:
        try:
            upload_box = st.container(key="ox_upload")
        except TypeError:
            upload_box = st.container()
        with upload_box:
            uploaded = st.file_uploader(UPLOAD_HELP, type=["png", "jpg", "jpeg"],
                                        label_visibility="collapsed", key=upload_key)
            st.markdown(f'<p class="ox-hint">{html_escape(UPLOAD_HELP)}</p>', unsafe_allow_html=True)
        if uploaded is not None:
            # detection showing: hide the upload box and its instructions to give the image the space
            st.markdown("<style>.st-key-ox_upload {display: none;}</style>", unsafe_allow_html=True)

            def _new_tile():
                st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
            st.button("Upload another tile", on_click=_new_tile)

    if uploaded is None:
        with ox_left:
            st.info("Upload a tile to run detection.")
    else:
        data = uploaded.getvalue()
        boxes, scores, size = detect(data)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        dets, tile_info, message = geolocate(uploaded.name, size, boxes, scores, threshold)

        with ox_left:
            shown, shown_dets = trim_nodata(image, dets)
            st.image(draw_detections(shown, shown_dets),
                     caption=f"{len(dets)} detection(s) at ≥ {threshold:.0%}",
                     use_container_width=True)
            with st.expander("Input"):
                st.image(image, caption="Input", use_container_width=True)
            if dets:
                def place_cell(d):
                    if d.get("place"):
                        text = d["place"]["text"]
                    elif d.get("center"):
                        text = fmt_latlon(*d["center"])
                    else:
                        text = "unknown"
                    return text + ("  [within 1 km of land, likely false]" if d["near_land"] else "")
                table = pd.DataFrame([{
                    "Confidence": f"{d['score']:.0%}",
                    "Place": place_cell(d),
                    "Coordinates": fmt_latlon(*d["center"]) if d.get("center") else "unknown",
                } for d in sorted(dets, key=lambda d: -d["score"])])
                table.index = range(1, len(table) + 1)
                # fixed, short height: many detections scroll inside the table, not the page
                st.dataframe(table, height=min(38 + 35 * len(table), 150),
                             use_container_width=True)
            else:
                st.info("No detections above the current threshold.")

        with ox_right:
            st.markdown("**Where is this tile?**")
            if tile_info is not None and tile_info.get("place"):
                st.caption(f"Tile centre: **{tile_info['place']['text']}** "
                           f"({fmt_latlon(*tile_info['center'])}).  Zoom out on the map to see our other detections.")
            # Fly-in animation once per uploaded file; later reruns (e.g. moving
            # the slider) redraw at the tile view without replaying it.
            key = hashlib.md5(data).hexdigest() + uploaded.name
            animate = st.session_state.get("animated_key") != key
            st.session_state["animated_key"] = key
            render_geomap(dets, tile_info, message, threshold, animate, height=MAP_HEIGHT)

        # Below the first screen, full width: sea floor and weather for this tile
        show_seafloor(tile_info)
        show_conditions(tile_info)

with tab2:
    centered_desc("SST, ERA5 wind and internal-wave detections together, June 2021. ERA5 and SST "
                  "are monthly data, so this is a monthly-representative view, not one day.")
    if not os.path.exists(SPATIAL_MATCHES_PATH):
        st.warning(f"Map data not found: place {SPATIAL_MATCHES_PATH} next to this script.")
    else:
        png, note = multimodal_map_png(files_stamp(
            SPATIAL_MATCHES_PATH, DETECTIONS_V2_PATH, DETECTIONS_OLD_PATH))
        st.image(png, use_container_width=True)
        st.caption(f"Detection spots: {note}.")