"""Shrink the detector checkpoint so it fits under GitHub's 100 MB file limit.

Run this where the checkpoint lives (e.g. a Kaggle notebook cell):

    !python shrink_checkpoint.py /kaggle/input/models/mayankgupta005/iw-detector-variants/pytorch/default/1/outputs/iw_improve/ckpt_improved_v2_s2.pth ckpt_improved_v2_s2.pth

It keeps only the model weights (drops optimizer state etc.) and stores them in
half precision. The dashboard copies them back into the float32 model on load,
so predictions are practically unchanged. The script prints how much the
detector's outputs shift, so you can check that yourself.
"""
import sys

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu")
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
small = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
torch.save({"model": small}, dst)


def build(sd):
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, min_size=1024, max_size=1024)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.load_state_dict(sd)
    return m.eval()


torch.manual_seed(0)
x = torch.rand(3, 1024, 1024)
with torch.no_grad():
    a = build(state)([x])[0]
    b = build(torch.load(dst, map_location="cpu")["model"])([x])[0]
import os
print(f"saved {dst}: {os.path.getsize(dst)/1e6:.1f} MB (original {os.path.getsize(src)/1e6:.1f} MB)")
print("detections on a random test image: full precision", len(a["boxes"]), "| half precision", len(b["boxes"]))
if len(a["boxes"]) and len(b["boxes"]):
    n = min(len(a["boxes"]), len(b["boxes"]))
    print("max box difference (px) among top", n, ":", float((a["boxes"][:n]-b["boxes"][:n]).abs().max()))
