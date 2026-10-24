# OceanXRay Internal-Wave Dashboard

Streamlit app: upload a Sentinel-1 tile, run the Faster R-CNN v2 internal-wave detector, and see the detected packets on the image and on a live map (with sea-floor depth, place names and scene-time weather).

This folder is the deployable version of the dashboard that used to be written out by `improve-landing.ipynb` and served through a temporary Cloudflare tunnel. Nothing needs the notebook any more.

## Files

| File | What it is |
|---|---|
| `iw_dashboard.py` | The app (entry point) |
| `oceanxray_geo.py`, `oceanxray_context.py`, `oceanxray_bathy.py`, `oceanxray_sea.py` | Helper modules (geolocation, place names + weather, GEBCO depth, ERA5 waves + sea temperature) |
| `requirements.txt` | Python packages (CPU-only PyTorch) |
| `tools/shrink_checkpoint.py` | Was used to shrink the checkpoint for GitHub (only needed if you retrain) |

## Data files (all included)

`data/` holds the detector checkpoint (`ckpt_improved_v2_s2.pth`, about 87 MB: weights only, stored in half precision), the geolocation table, the v2 detections, the land/ocean mask and the SST/wind matches. `oceanxray_bathy_grid.bin`, `oceanxray_context_data.bin` and `oceanxray_sea_data.bin` (about 0.9 MB, offline June 2021 waves and sea temperature) sit next to `iw_dashboard.py`. If any optional file is missing the matching panel is hidden or shows a notice; only the checkpoint is required.

The half-precision checkpoint was compared with the original on real tiles: same boxes at the app's 0.7 threshold, box positions within 1 px, scores within 0.004.

## Deploy on Streamlit Community Cloud (free)

1. Push this whole folder to a **new GitHub repo** (suggestion: `oceanxray-dashboard`). A separate repo keeps your notebook repo exactly as the reviewers will see it. The checkpoint is under GitHub's 100 MB limit (GitHub only warns above 50 MB).
2. Go to share.streamlit.io, sign in with GitHub, choose **Create app**, pick the new repo, and set the main file to `iw_dashboard.py`. Under **Advanced settings** choose Python 3.12.
3. Deploy. The first build downloads PyTorch and takes several minutes. You get a permanent `*.streamlit.app` link.

Notes:
- Free apps go to sleep after a period with no visitors; opening the link wakes them.
- The free tier has limited memory (about 2.7 GB according to Streamlit's docs; in my test the app peaked near 2.3 GB). Detection runs on CPU at 1024 px, so each tile takes seconds, not milliseconds. If the app crashes with an out-of-memory error, that is the cause; moving to a paid host with more RAM fixes it.
- If the coastline download for the multimodal map fails, the map falls back to a plain background instead of crashing.
- The live map loads its tiles from Esri and CARTO, and weather from Open-Meteo, so the visitor's browser and the server both need internet access (normal on Streamlit Cloud).

## Run locally

```bash
pip install -r requirements.txt
streamlit run iw_dashboard.py
```

To keep the data somewhere else, set `OCEANXRAY_DATA` to that folder.
