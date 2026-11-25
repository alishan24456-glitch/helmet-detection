#!/usr/bin/env python3
"""
Helmet Detection (traffic enforcement)
========================================

A roadside / traffic-camera helmet check for motorcycle riders.
YOLOv8n finds motorcycles and riders (persons), each person is
associated with a motorcycle (overlap / sitting above the bike
box), and the head region of each rider -- the top ~25% of the
person box -- is checked for a helmet, giving a HELMET /
NO HELMET verdict per rider:

* **Head region** -- the top quarter of the rider's person box,
  inset 15% on the sides. It is a documented IMAGE REGION for
  this scene, not a calibrated camera model: on a rider the head
  sits at the top of the person box. A pillion rider's head is
  smaller and partly hidden behind the driver, so it reads
  UNCLEAR more often (see the limits below).
* **Helmet check** -- a helmet is a smooth rounded shell: one
  solid colour covering the whole top of the head. A bare head
  is hair on top and face below. The check measures two things
  on the head crop: (1) the shell blob -- the crop is blurred,
  quantised to coarse colour bins, and the largest CONNECTED
  blob of the dominant colour is measured in the CENTRAL band
  of the crop (the middle 60% of its width, where the head
  sits -- the full crop's dominant blob is often the
  background, trees or a neighbouring rider, which read as a
  dark "shell" and caused false NO HELMETs on the fourth
  verification pass); a shell is a big bright blob at the top
  of the band (red/white shells read blob 0.47-0.56, V 220-235
  in simulation); (2) the skin fraction (HSV skin range): a
  bare face fills the crop with skin (0.45 in simulation), a
  helmeted head shows only a sliver of face (0.04-0.05).
  HELMET needs a bright top blob >= 0.30; NO HELMET needs
  skin >= 0.45; anything else, or a head too small / too dark
  / clipped by the frame edge, reads UNCLEAR. The thresholds
  were set from measured head crops on the demo clip and in
  simulation (see README) and are deliberately conservative:
  a verdict is a flag FOR HUMAN REVIEW, never a fine. Two
  limits are accepted outright: a DARK helmet reads UNCLEAR
  (its shell is as dark as hair, so brightness cannot tell
  them apart -- the check refuses to guess), and a close rider
  whose face fills the crop can read NO HELMET on skin alone
  (the bar is 0.45, above the highest helmeted-face reading
  measured on the demo clip, 0.42). (Earlier builds also tried
  a dark-top-blob NO HELMET path and edge density: the dark
  blob misread dark trees and dark helmets as hair, and edge
  density cannot separate the two either -- on the demo clip
  every helmeted head read edge 0.11-0.26, the same range as
  a bare head. Edge density is logged only as a diagnostic.)
* **Confirmation** -- a rider verdict must repeat in 3
  consecutive detection rounds before it counts and is logged,
  so one flickering YOLO box or one glinting visor cannot raise
  a flag.

The compliance board shows riders tracked, HELMET / NO HELMET /
UNCLEAR counts and the compliance rate among decided riders
(helmet / (helmet + no helmet)). A NO HELMET rider raises a red
banner and a timestamped event-log entry. A no-helmet rider can
be linked to the bike's plate with the license-plate-recognition
project's code (same repo family); this script does not read
plates itself.

Weights: ``yolov8n.onnx`` next to this script (12.8 MB, ships in
this folder). If it is missing, the script tries to export it from
``yolov8n.pt`` with Ultralytics (downloading the .pt if needed) and
otherwise exits with instructions.

Simulation mode (``--simulate``) renders a roadside view with
three motorcycles: bike 1 has a helmeted rider, bike 2 a bareheaded
rider, bike 3 a helmeted driver with a bareheaded pillion. The
simulation's person/motorcycle source finds the bikes and riders
with colour masks + contours -- the head region, the helmet check
and the verdict logic are the exact same code as real mode, and
the detector never sees the simulation's ground truth. Simulation
is an internal logic check; the delivered demo is real footage
(see README).

Honest framing: a helmet verdict is a flag for human review,
never a fine. Small/distant heads and pillion riders can be
missed, a dark helmet on dark hair can read NO HELMET, and a
bare head in shadow can read UNCLEAR -- the board counts what
the pixels support, nothing more.

Examples
--------
    python helmet_detection.py                 (file picker)
    python helmet_detection.py --source pick
    python helmet_detection.py --source road.mp4
    python helmet_detection.py --source 0      (webcam)
    python helmet_detection.py --simulate --max-seconds 30 --no-display

Press Q to quit. The video is written to ``output.mp4``, the
timestamped event log to ``output_events.log`` and the final
counters to ``output_summary.txt`` next to it.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Layout (the output video is always 1280x720)
# ---------------------------------------------------------------------------
W, H = 1280, 720
TITLE_H = 46
PANEL_H = 150                     # stats panel at the bottom
VIDEO_W, VIDEO_H = W, H - TITLE_H - PANEL_H
BG = (24, 26, 32)
PANEL_BG = (32, 35, 46)
WHITE = (255, 255, 255)
GREY = (160, 166, 178)
GREEN = (90, 200, 90)
AMBER = (60, 170, 255)
RED = (60, 60, 235)
YELLOW = (0, 210, 235)

# ---------------------------------------------------------------------------
# Detection + helmet-check tuning
# ---------------------------------------------------------------------------
DET_W, DET_H = 1280, 720          # detection runs on this frame size
DETECT_EVERY = 3                  # run YOLO every 3rd frame (10 fps)
CONF = 0.30
NMS_THRESH = 0.30
PERSON_CLASS = 0                  # COCO person
MOTORCYCLE_CLASS = 3              # COCO motorcycle
MATCH_DIST = 80                   # px, centroid match radius
MAX_MISSED = 15                   # detection rounds before a track drops
# Head region: top fraction of the person box, inset on the sides.
HEAD_TOP_FRAC = 0.25
HEAD_SIDE_INSET = 0.15
MIN_HEAD_W, MIN_HEAD_H = 18, 18  # smaller than this: UNCLEAR
MIN_MEAN_BRIGHTNESS = 40.0        # darker than this: UNCLEAR, not NO HELMET
# Helmet check thresholds (measured on the demo clip + simulation;
# see README). Simulation, central band: helmeted shell blob
# 0.47-0.56, V 220-235, skin 0.05; bare skin 0.45. Real helmeted
# heads on the demo clip: blob 0.09-0.60, skin 0.00-0.13 -- the
# crowded queue fragments the shell blob, so many real riders
# read UNCLEAR rather than HELMET; that is the honest result,
# not a failure to detect. A dark helmet (V 40-100) reads
# UNCLEAR by design -- its shell is as dark as hair.
BLOB_MIN = 0.30                   # a top cover this big decides
BLOB_BRIGHT_MIN = 65.0            # blob mean V at/above: a shell
                                  # (hair reads V ~35; dark helmets
                                  # V ~40-100, so a dark shell
                                  # reads UNCLEAR, never NO HELMET)
SKIN_BARE_MIN = 0.45              # at or above: clearly a bare face
                                  # (highest helmeted-face reading
                                  # on the demo clip: 0.42)
CONFIRM_ROUNDS = 3                # a verdict must repeat this often
                                  # before it counts and is logged
PT_URL = ("https://github.com/ultralytics/assets/releases/"
          "download/v8.2.0/yolov8n.pt")


# ---------------------------------------------------------------------------
# Rider sources -- real mode (YOLOv8 via OpenCV DNN) and simulation
# (colour masks). Both return {"persons": [...], "motorcycles": [...]}
# with boxes [x, y, w, h, conf] in detection-frame coordinates.
# ---------------------------------------------------------------------------
def ensure_weights(path):
    """Return a usable yolov8n.onnx path, exporting it if needed."""
    if os.path.exists(path):
        return path
    print(f"Weights not found at {path}; trying Ultralytics export ...")
    try:
        from ultralytics import YOLO
        pt = os.path.join(os.path.dirname(path) or ".", "yolov8n.pt")
        if not os.path.exists(pt):
            print("Downloading yolov8n.pt ...")
            urllib.request.urlretrieve(PT_URL, pt)
        print("Exporting yolov8n.pt -> yolov8n.onnx (one-time) ...")
        YOLO(pt).export(format="onnx", imgsz=640)
        exported = os.path.join(os.path.dirname(pt) or ".", "yolov8n.onnx")
        if os.path.exists(exported) and exported != path:
            os.replace(exported, path)
        return path
    except Exception as exc:  # noqa: BLE001 - report and stop cleanly
        print(f"Could not obtain weights: {exc}")
        print("Download yolov8n.onnx and place it next to this script, or")
        print("install ultralytics (`pip install ultralytics`) and re-run.")
        sys.exit(1)


class YoloRiderDetector:
    """YOLOv8n person + motorcycle detection through OpenCV DNN."""

    def __init__(self, weights_path):
        self.net = cv2.dnn.readNetFromONNX(ensure_weights(weights_path))

    def detect(self, frame):
        """frame is a DET_W x DET_H image; returns person/bike boxes."""
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()[0].T  # [8400, 84]: cx cy w h + 80 scores
        # NOTE: YOLOv8 has NO objectness channel -- confidence is the
        # max class score and the class is its argmax (a v7-style
        # objectness read would silently kill every detection).
        scores = out[:, 4:].max(axis=1)
        classes = out[:, 4:].argmax(axis=1)
        keep = (scores > CONF) & np.isin(classes, [PERSON_CLASS,
                                                   MOTORCYCLE_CLASS])
        sh, sw = frame.shape[:2]
        boxes = []
        for (cx, cy, bw, bh), s, c in zip(out[keep, :4], scores[keep],
                                         classes[keep]):
            boxes.append([float((cx - bw / 2) * sw / 640),
                          float((cy - bh / 2) * sh / 640),
                          float(bw * sw / 640), float(bh * sh / 640),
                          float(s), int(c)])
        if not boxes:
            return {"persons": [], "motorcycles": []}
        idx = cv2.dnn.NMSBoxes([b[:4] for b in boxes],
                               [b[4] for b in boxes], CONF, NMS_THRESH)
        kept = [boxes[int(i)] for i in idx.flatten()] if len(idx) else []
        return {"persons": [b for b in kept if b[5] == PERSON_CLASS],
                "motorcycles": [b for b in kept
                                if b[5] == MOTORCYCLE_CLASS]}


def _mask_boxes(frame, lo, hi, min_area):
    """Boxes of one colour family, from an HSV mask + contours."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([float(x), float(y), float(w), float(h), 1.0])
    return boxes


def sim_rider_boxes(frame):
    """Colour-blob person/motorcycle source for the rendered roadside.

    Motorcycles are drawn in crowd-density blue. A rider is drawn
    in saturated tan skin, with a solid-colour helmet shell where
    helmeted and a dark hair cap where bare. No single mask covers
    a whole rider -- the skin mask alone sees only the face and
    torso, so its box starts at the jaw and the head crop then
    sits too high (on a bare rider it covered 19 px of uniform
    road above the head, which read as one big shell blob -- a
    false HELMET on the third simulation run; growing the box by
    a fixed percentage overshot the bare riders by the same
    amount). The person mask is therefore the UNION of four
    masks -- tan skin, saturated red (red helmet shell), dark
    (hair cap, visor, wheels) and bright white restricted to the
    sky band (white helmet shell; the road dashes are bright too
    but sit below the sky band, and the sky itself is V 190,
    below the 200 floor). The union's bounding box covers the
    whole rider including the head, the way a YOLO person box
    does in real mode -- and the helmet verdict still comes only
    from the pixels inside the head crop.
    """
    bikes = _mask_boxes(frame, [95, 120, 80], [115, 255, 255], 400)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    tan = cv2.inRange(hsv, np.array([5, 100, 150]),
                      np.array([35, 220, 255]))
    red = cv2.inRange(hsv, np.array([0, 150, 150]),
                      np.array([8, 255, 255]))
    dark = cv2.inRange(hsv, np.array([0, 0, 0]),
                       np.array([180, 120, 60]))
    bright = cv2.inRange(hsv, np.array([0, 0, 200]),
                         np.array([180, 60, 255]))
    bright[int(DET_H * 0.45):, :] = 0  # sky band only (see above)
    mask = cv2.bitwise_or(tan, cv2.bitwise_or(red, dark))
    mask = cv2.bitwise_or(mask, bright)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    persons = []
    for c in contours:
        if cv2.contourArea(c) < 300:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # The wheels are dark blobs of their own, well below any
        # rider; a rider box must sit above them. (The check
        # lives here, in the sim source, because the union mask
        # deliberately includes dark wheels. The floor is 0.60H:
        # the lowest rider top is 393 px on the far lane, the
        # highest wheel top is 468 px.)
        if y > DET_H * 0.60:
            continue
        persons.append([float(x), float(y), float(w), float(h), 1.0])
    return {"persons": persons, "motorcycles": bikes}


# ---------------------------------------------------------------------------
# Rider association + helmet check -- the same code on real and
# simulated pixels
# ---------------------------------------------------------------------------
def associate_riders(persons, motorcycles):
    """Pair each person box with a motorcycle box.

    A person is a rider of a bike when the boxes overlap, or when
    the person's centre-x lies inside the bike's x-range (expanded
    10%) and the person's bottom sits between the bike's top and
    its bottom -- a rider sits ON the bike, so the person box
    overlaps the bike box's upper part. Returns a list of
    (person_box, bike_box_or_None); a person with no bike is kept
    with None (a pedestrian, or a rider whose bike YOLO missed)
    and is NOT judged -- a helmet check needs a bike.
    """
    pairs = []
    for p in persons:
        px, py, pw, ph, _ = p[:5]
        pcx, pbottom = px + pw / 2, py + ph
        best, best_score = None, 0.0
        for b in motorcycles:
            bx, by, bw, bh, _ = b[:5]
            ix0, iy0 = max(px, bx), max(py, by)
            ix1, iy1 = min(px + pw, bx + bw), min(py + ph, by + bh)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            overlap = inter / (pw * ph) if pw * ph > 0 else 0.0
            sits_above = (bx - 0.10 * bw <= pcx <= bx + 1.10 * bw
                          and by <= pbottom <= by + bh)
            score = overlap + (0.5 if sits_above else 0.0)
            if score > best_score:
                best, best_score = b, score
        # A rider must actually touch the bike: at least 2% of the
        # person box overlapping it, or sitting above it. A
        # pedestrian standing next to a parked bike scores 0 and
        # stays unpaired.
        pairs.append((p, best if best_score >= 0.02 else None))
    return pairs


def head_rect(person_box):
    """Head rectangle: top HEAD_TOP_FRAC of the person box.

    Returns (x0, y0, x1, y1) in detection-frame coordinates, or
    None when the head is too small to judge or clipped by the
    frame edge.
    """
    x, y, w, h, _ = person_box[:5]
    # 50 px, not 60: the simulated riders' person boxes are
    # 54-57 px wide after the arm fix, and a 54 px box still
    # yields a 38x33 px head crop -- judgeable. (At 60 the
    # fourth simulation run judged zero riders.)
    if w < 50 or h < 50:
        return None
    # A rider sliding into frame has a clipped person box: the
    # head crop is then a partial dome against uniform sky, which
    # reads as one big shell blob (a false HELMET at 5.2 s in the
    # first simulation run) or a partial face (a false NO HELMET
    # at 1.1 s). A clipped rider reads UNCLEAR until fully in
    # frame -- the check is on the PERSON box, not just the head
    # rect, because the head rect itself can sit clear of the
    # edge while the rider is still half out of frame.
    if x <= 2 or y <= 2 or x + w >= DET_W - 2 or y + h >= DET_H - 2:
        return None
    hx0 = x + HEAD_SIDE_INSET * w
    hx1 = x + (1 - HEAD_SIDE_INSET) * w
    hy0, hy1 = y, y + HEAD_TOP_FRAC * h
    if hx1 - hx0 < MIN_HEAD_W or hy1 - hy0 < MIN_HEAD_H:
        return None
    return (hx0, hy0, hx1, hy1)


def helmet_check(frame, rect):
    """Verdict for one rider's head: 'HELMET', 'NO HELMET' or 'UNCLEAR'.

    ``frame`` is the BGR detection frame; ``rect`` is the head
    rectangle. Returns (verdict, score, detail) where score is a
    0-1 helmet-likeness (high = helmet) and detail is a short
    string for the event log / debugging.
    """
    x0, y0, x1, y1 = [int(v) for v in rect]
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return "UNCLEAR", 0.0, "empty head crop"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean < MIN_MEAN_BRIGHTNESS:
        return "UNCLEAR", 0.0, f"too dark (mean {mean:.0f})"
    # (1) Shell blob, measured in the CENTRAL band of the crop
    # (the middle 60% of its width): the head sits in the centre
    # of the person box, while the full crop's dominant blob is
    # often the background -- dark trees beside the head read as
    # a dark "shell" and caused false NO HELMETs. Blur away
    # shading gradients, quantise to coarse colour bins, and
    # measure the largest CONNECTED component of the dominant
    # colour, with its mean brightness (V) and its vertical
    # centre. A helmet shell is one big bright blob at the top
    # of the band; a bare head's top cover is hair -- also one
    # big blob, but near-black, so it reads UNCLEAR on
    # brightness, never HELMET. (The blur matters: without it a
    # dome's shading gradient splits it into several bins and
    # the blob collapses -- measured on the demo clip. The
    # centroid check matters too: without it a bare face filling
    # the bottom of the crop could pose as a "shell".)
    cw = crop.shape[1]
    band = crop[:, int(cw * 0.2):int(cw * 0.8)]
    blur = cv2.GaussianBlur(band, (5, 5), 0)
    quant = blur // 64
    flat = quant.reshape(-1, 3)
    vals, counts = np.unique(flat, axis=0, return_counts=True)
    dom = float(counts.max() / flat.shape[0])
    top = vals[counts.argmax()]
    mask = np.all(quant == top, axis=2).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, 8)
    blob = 0.0
    blob_v = 0.0
    blob_cy = 1.0
    biggest = None
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        blob = float(stats[biggest, cv2.CC_STAT_AREA] / mask.size)
        blob_cy = float(centroids[biggest][1] / mask.shape[0])
    # Mean V of the blob's pixels, from the unblurred band.
    hsv_band = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    if biggest is not None:
        blob_v = float(hsv_band[:, :, 2][labels == biggest].mean())
    # Skin fraction is measured on the FULL crop (the face sits
    # low in it, under the shell).
    hsv_full = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # (2) Skin fraction: a bare face fills the head crop with
    # skin; a helmeted head shows only a sliver of face under
    # the shell. The range is deliberately wide (H 0-30, S 30+)
    # so profile faces in warm light still count -- but S is
    # capped at 180: a saturated red helmet (S ~208) sits in
    # the same hue band as skin, and without the cap it counted
    # as "skin" and a red-helmeted rider read NO HELMET (found
    # on the second simulation run; the drawn skin tone is
    # S ~146, safely inside the cap).
    skin = cv2.inRange(hsv_full, np.array([0, 30, 40]),
                       np.array([30, 180, 255]))
    skin_frac = float((skin > 0).mean())
    # Diagnostic only: Canny edge density. It does NOT decide
    # the verdict -- on the demo clip helmeted heads read
    # 0.11-0.26 (visor edges, vents, neighbouring riders), the
    # same range as a bare head, so it cannot separate them.
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    score = blob
    detail = (f"blob {blob:.2f} V{blob_v:.0f} skin {skin_frac:.2f} "
              f"dom {dom:.2f} edge {edge_density:.2f}")
    # A dark top blob (hair, or a dark helmet, or dark
    # background) reads UNCLEAR -- never NO HELMET. Only a
    # clearly bare face, skin filling nearly half the crop,
    # raises the flag.
    top_cover = blob >= BLOB_MIN and blob_cy < 0.5
    if top_cover and blob_v >= BLOB_BRIGHT_MIN:
        return "HELMET", score, detail
    if skin_frac >= SKIN_BARE_MIN:
        return "NO HELMET", 1.0 - skin_frac, detail
    return "UNCLEAR", score, detail


# ---------------------------------------------------------------------------
# Helmet tracker -- centroid tracking + per-rider confirmed verdicts
# ---------------------------------------------------------------------------
class HelmetTracker:
    """Tracks riders (person + bike pairs) and their helmet verdicts."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.frames = 0
        self.video_frames = 0
        self.events = []
        self.person_detections = 0
        self.motorcycle_detections = 0
        self.riders_seen = 0
        self._last_t = None
        # Counted rider outcomes, keyed by rider id: the verdict a
        # rider counts as. A rider counts its FIRST decided verdict
        # (HELMET / NO HELMET); a rider that first reads UNCLEAR
        # while sliding into frame and is later decided counts the
        # decided verdict, not the early UNCLEAR. A decided rider
        # that later goes unreadable keeps its decided outcome (the
        # log still records the change); a flip between decided
        # verdicts moves the count.
        self.rider_outcomes = {}
        self.no_helmet_events = 0
        self.first_no_helmet_t = None
        # Head-crop stats of confirmed verdicts, for the README:
        # (verdict, edge density, skin fraction) sampled at confirm.
        self.head_stats = []

    @property
    def helmet(self):
        return sum(1 for v in self.rider_outcomes.values()
                   if v == "HELMET")

    @property
    def no_helmet(self):
        return sum(1 for v in self.rider_outcomes.values()
                   if v == "NO HELMET")

    @property
    def unclear(self):
        return sum(1 for v in self.rider_outcomes.values()
                   if v == "UNCLEAR")

    def _log(self, t, text):
        self.events.append((t, text))

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a[:4]
        bx, by, bw, bh = b[:4]
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _centre(box):
        x, y, w, h = box[:4]
        return (x + w / 2, y + h / 2)

    def reset(self, t):
        """Drop all tracks: the source looped, so the scene restarted."""
        n = len(self.tracks)
        self.tracks.clear()
        self._log(t, f"source looped -- {n} in-flight tracks reset "
                     f"(rider verdicts restart from zero)")

    @staticmethod
    def _new_rider():
        return {"pending": None, "streak": 0, "confirmed": None,
                "score": 0.0, "detail": ""}

    def update(self, pairs, frame, t):
        """One detection round. Returns the live state dict."""
        self.frames += 1
        self._last_t = t
        used = set()
        for person, bike in pairs:
            if bike is None:
                continue  # not a rider: no bike, no verdict
            cx, cy = self._centre(person)
            best, best_d = None, float("inf")
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                d = math.hypot(cx - tr["cx"], cy - tr["cy"])
                gate = max(MATCH_DIST,
                           0.5 * max(person[2], person[3],
                                     tr["person"][2], tr["person"][3]))
                if d >= gate:
                    continue
                if d >= best_d:
                    continue
                # A detection must also OVERLAP the track's last
                # person box (low floor: a fast-approaching rider
                # grows quickly between rounds).
                if self._iou(person, tr["person"]) <= 0.02:
                    continue
                best, best_d = tid, d
            if best is None:
                tid = self.next_id
                self.next_id += 1
                self.riders_seen += 1
                self.tracks[tid] = {"id": tid, "cx": cx, "cy": cy,
                                    "person": person, "bike": bike,
                                    "missed": 0,
                                    "rider": self._new_rider()}
                best = tid
            used.add(best)
            tr = self.tracks[best]
            tr.update({"cx": cx, "cy": cy, "person": person,
                       "bike": bike, "missed": 0})
            rider = tr["rider"]
            rect = head_rect(person)
            if rect is None:
                verdict, score, detail = (
                    "UNCLEAR", 0.0, "head too small / clipped")
            else:
                verdict, score, detail = helmet_check(frame, rect)
            if verdict == rider["pending"]:
                rider["streak"] += 1
            else:
                rider["pending"] = verdict
                rider["streak"] = 1
            rider["score"] = score
            rider["detail"] = detail
            if (rider["streak"] >= CONFIRM_ROUNDS
                    and rider["confirmed"] != verdict):
                prev = rider["confirmed"]
                rider["confirmed"] = verdict
                key = best
                prev_outcome = self.rider_outcomes.get(key)
                if verdict in ("HELMET", "NO HELMET"):
                    if prev_outcome != verdict:
                        if verdict == "NO HELMET":
                            self.no_helmet_events += 1
                            if self.first_no_helmet_t is None:
                                self.first_no_helmet_t = t
                        self.rider_outcomes[key] = verdict
                    # Record the head-crop stats at confirmation.
                    # detail reads "blob X VY skin Z dom W edge V".
                    try:
                        parts = detail.split()
                        blob = float(parts[1])
                        skin = float(parts[4])
                        self.head_stats.append((verdict, blob, skin))
                    except (IndexError, ValueError):
                        pass
                elif prev_outcome is None:
                    self.rider_outcomes[key] = "UNCLEAR"
                # NOTE: log `best`, the matched track id -- NOT
                # `tid`, which is only assigned when a NEW track
                # is created and otherwise keeps the id of the
                # last-created track. Logging `tid` here printed
                # the wrong rider on later transitions (the
                # second simulation run logged rider #1's exit as
                # rider #2 and rider #3's exit as rider #4) while
                # the counts -- keyed by `best` -- stayed right.
                if verdict == "NO HELMET":
                    self._log(t, f"rider #{best} NO HELMET -- flagged "
                                 f"for human review ({detail})")
                elif verdict == "HELMET":
                    self._log(t, f"rider #{best} HELMET ({detail})")
                elif prev is not None:
                    self._log(t, f"rider #{best} now UNCLEAR "
                                 f"(was {prev})")
        for tid in list(self.tracks):
            if tid in used:
                continue
            tr = self.tracks[tid]
            tr["missed"] += 1
            if tr["missed"] > MAX_MISSED:
                if tr["rider"]["confirmed"] == "NO HELMET":
                    self._log(t, f"rider #{tid} left the frame "
                                 f"(NO HELMET flagged)")
                del self.tracks[tid]
        # Live board state.
        any_no_helmet = any(
            tr["rider"]["confirmed"] == "NO HELMET"
            for tr in self.tracks.values())
        decided = self.helmet + self.no_helmet
        compliance = (100.0 * self.helmet / decided) if decided else None
        return {"alert": ("NO HELMET FLAGGED"
                          if any_no_helmet else "MONITORING"),
                "any_no_helmet": any_no_helmet,
                "compliance": compliance}

    def summary(self):
        decided = self.helmet + self.no_helmet
        lines = [
            "HELMET DETECTION -- RUN SUMMARY",
            f"frames processed: {self.video_frames}",
            f"tracker updates: {self.frames}",
            f"person detections (summed): {self.person_detections}",
            f"motorcycle detections (summed): "
            f"{self.motorcycle_detections}",
            f"riders tracked: {self.riders_seen}",
            f"riders assessed (confirmed verdicts): "
            f"{self.helmet + self.no_helmet + self.unclear}",
            f"  HELMET: {self.helmet}",
            f"  NO HELMET (flagged for human review): {self.no_helmet}",
            f"  UNCLEAR: {self.unclear}",
            ("compliance among decided riders: "
             + (f"{100.0 * self.helmet / decided:.1f}% "
                f"({self.helmet}/{decided})" if decided
                else "no decided riders")),
            f"NO HELMET events logged: {self.no_helmet_events}",
            ("first NO HELMET at: "
             + (f"{self.first_no_helmet_t:.1f} s"
                if self.first_no_helmet_t is not None else "never")),
            f"events logged: {len(self.events)}",
        ]
        if self.head_stats:
            for verdict in ("HELMET", "NO HELMET"):
                rows = [r for r in self.head_stats if r[0] == verdict]
                if rows:
                    blobs = [r[1] for r in rows]
                    skins = [r[2] for r in rows]
                    lines.append(
                        f"head crops {verdict}: n={len(rows)} "
                        f"shell blob {min(blobs):.2f}-"
                        f"{max(blobs):.2f} "
                        f"skin fraction {min(skins):.2f}-"
                        f"{max(skins):.2f}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Simulation renderer -- a roadside view; motorcycles ride past with
# helmeted and bareheaded riders. The tracker only ever sees these
# pixels.
# ---------------------------------------------------------------------------
class RoadsideSim:
    """Renders motorcycles riding across a road, riders on top.

    The drawn bikes use the geometry the pipeline assumes: the
    rider sits above the bike box, so the person box overlaps the
    bike box's upper part. A helmeted rider gets a solid-colour
    rounded dome over the head; a bareheaded rider gets a skin
    head with a hair cap and a face line. The helmet check has to
    find the difference in the rendered pixels -- it never sees
    the script's ground truth.
    """

    BIKE_BLUE = (255, 130, 30)    # crowd-density blue: the bike mask
    BIKE_DARK = (200, 100, 20)    # darker blue, still inside the mask
    SKIN = (90, 140, 210)         # rider skin: a saturated tan (S ~146)
                                  # in the tan mask -- the first skin tone
                                  # (170,190,220; S ~58) sat inside the
                                  # sky's HSV range, so the mask caught
                                  # the whole sky as one "person"
    SHIRT = (60, 90, 180)         # rider shirt -- NOT tan, so the tan
                                  # mask sees the torso only via the skin
                                  # arms; see _draw_rider note below
    HAIR = (30, 30, 35)           # bare head: dark hair cap + face line
    HELMET_RED = (40, 60, 220)    # solid saturated helmet shell
    HELMET_WHITE = (235, 235, 235)

    def __init__(self, w, h, fps=30):
        self.w, self.h = w, h
        self.fps = fps
        self.frame_idx = 0
        # (lane-y frac, enter t, leave t, speed px/s, helmeted?,
        #  pillion helmeted?/None for no pillion, helmet colour).
        self.bikes = [
            {"y": 0.62, "enter": 0.0, "leave": 12.0, "speed": 90,
             "helmet": True, "pillion": None,
             "colour": self.HELMET_RED},
            {"y": 0.78, "enter": 4.0, "leave": 18.0, "speed": 110,
             "helmet": False, "pillion": None,
             "colour": None},
            {"y": 0.62, "enter": 16.0, "leave": 28.0, "speed": 90,
             "helmet": True, "pillion": False,
             "colour": self.HELMET_WHITE},
        ]

    def _draw_road(self, img):
        # Pale grey-blue sky (BGR 190,170,150; S ~54): the
        # previous sky (200,160,100; S ~127) sat inside the
        # blue motorcycle mask, so the whole sky was detected
        # as one giant "motorcycle" (~2.2 bike boxes/frame in
        # the first run). The first sky (120,140,165; hue 13,
        # S 70) sat inside the skin-colour range instead, so
        # sky pixels inside a head crop counted as "skin" and
        # a white-helmeted rider read skin 0.79.
        cv2.rectangle(img, (0, 0), (self.w, int(self.h * 0.45)),
                      (190, 170, 150), -1)
        cv2.rectangle(img, (0, int(self.h * 0.45)), (self.w, self.h),
                      (70, 72, 76), -1)
        for x in range(0, self.w, 90):
            cv2.circle(img, (x + 20, int(self.h * 0.42)), 24,
                       (60, 110, 60), -1)
        for i in range(4):
            y = int(self.h * (0.52 + i * 0.11))
            cv2.rectangle(img, (self.w // 2 - 6, y),
                          (self.w // 2 + 6, y + 26), (220, 220, 210), -1)

    def _draw_rider(self, img, cx, top, h, helmeted, colour):
        """One rider figure: head + torso above the bike.

        The figure is drawn in SKIN (tan) so the tan person mask
        finds it: head circle, neck and a torso block, all skin.
        A helmeted rider gets a solid-colour dome drawn OVER the
        head -- the dome is not tan, so the person box comes from
        the torso and the lower head, exactly as a real YOLO
        person box still covers a helmeted rider. A bareheaded
        rider gets a dark hair cap over the top of the head and a
        short face line, so the head crop carries hair texture
        and skin.
        """
        head_r = max(8, int(h * 0.16))
        head_cy = top + head_r
        # Torso: from just below the head to the bike seat.
        cv2.rectangle(img, (cx - head_r, head_cy + head_r // 2),
                      (cx + head_r, top + h), self.SKIN, -1)
        # Arm reaching forward/down to the handlebars. It stays
        # inside the torso silhouette (only 8 px past it): the
        # first version reached 26 px past, which widened the
        # colour-mask person box well past the head, so the head
        # crop was centred on the box instead of the head and
        # filled with uniform road -- a false HELMET blob on the
        # fourth simulation run.
        cv2.line(img, (cx + head_r // 2, head_cy + head_r),
                 (cx + head_r + 8, top + h - 6), self.SKIN, 7)
        cv2.circle(img, (cx, head_cy), head_r, self.SKIN, -1)
        if helmeted:
            # Solid dome: covers the top ~70% of the head circle
            # and sticks out a little wider than the head, like a
            # real shell. Drawn as a filled ellipse + a visor band.
            cv2.ellipse(img, (cx, head_cy - head_r // 4),
                        (head_r + 4, head_r + 2), 0, 180, 360,
                        colour, -1)
            cv2.rectangle(img, (cx - head_r - 2, head_cy - 2),
                          (cx + head_r + 2, head_cy + head_r // 3),
                          (30, 30, 30), -1)
        else:
            # Hair cap over the top half of the head + a face line.
            cv2.ellipse(img, (cx, head_cy - head_r // 4),
                        (head_r + 1, head_r // 2 + 1), 0, 180, 360,
                        self.HAIR, -1)
            cv2.line(img, (cx - head_r // 2, head_cy + head_r // 3),
                     (cx + head_r // 2, head_cy + head_r // 3),
                     self.HAIR, 2)

    def _draw_bike(self, img, cx, ground_y, helmeted, pillion, colour):
        # Big enough for the head check: at 150 px wide the rider's
        # person box came out 54 px wide and head_rect() rejected
        # it as too small -- the first simulation run judged zero
        # riders. At 240 px the rider is ~90x150, like a mid-range
        # rider on the real clip.
        bw, bh = 240, 100
        x0, y0 = int(cx - bw / 2), int(ground_y - bh)
        x1, y1 = int(cx + bw / 2), int(ground_y)
        # Wheels.
        cv2.circle(img, (x0 + 40, y1 - 18), 30, (20, 20, 22), -1)
        cv2.circle(img, (x1 - 40, y1 - 18), 30, (20, 20, 22), -1)
        cv2.circle(img, (x0 + 40, y1 - 18), 14, (90, 90, 95), -1)
        cv2.circle(img, (x1 - 40, y1 - 18), 14, (90, 90, 95), -1)
        # Body: tank + seat, in the bike-blue mask range.
        cv2.rectangle(img, (x0 + 28, y0 + 28), (x1 - 28, y0 + 64),
                      self.BIKE_BLUE, -1)
        cv2.rectangle(img, (x0 + 70, y0 + 10), (x0 + 134, y0 + 34),
                      self.BIKE_DARK, -1)
        cv2.line(img, (x1 - 40, y0 + 28), (x1 - 12, y0 - 10),
                 self.BIKE_BLUE, 9)
        # Driver sits above the seat; pillion behind and higher.
        # The pillion sits 90 px behind the driver: at 52 px the
        # two tan figures touched and the colour mask merged them
        # into one person box, so the pillion was never judged.
        rider_h = 130
        self._draw_rider(img, cx + 10, y0 - rider_h + 22, rider_h,
                         helmeted, colour)
        if pillion is not None:
            self._draw_rider(img, cx - 80, y0 - rider_h + 4, rider_h,
                             pillion, self.HELMET_RED)

    def frame(self):
        idx = self.frame_idx
        self.frame_idx += 1
        t = idx / self.fps
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._draw_road(img)
        for bike in self.bikes:
            if t < bike["enter"] or t > bike["leave"]:
                continue
            span = bike["leave"] - bike["enter"]
            f = (t - bike["enter"]) / max(1.0, span)
            cx = -120 + f * (self.w + 240)
            ground_y = int(self.h * bike["y"]) + 40
            self._draw_bike(img, int(cx), ground_y, bike["helmet"],
                            bike["pillion"], bike["colour"])
        return img


# ---------------------------------------------------------------------------
# Rendering -- draws tracker output onto the 1280x720 canvas
# ---------------------------------------------------------------------------
def draw_canvas(det_frame, tracker, res, elapsed, mode_label):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = BG
    cv2.rectangle(canvas, (0, 0), (W, TITLE_H), PANEL_BG, -1)
    cv2.putText(canvas, "HELMET DETECTION  --  motorcycle riders",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    cv2.putText(canvas, mode_label, (W - 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)

    vh, vw = det_frame.shape[:2]
    scale = min(VIDEO_W / vw, VIDEO_H / vh)
    nw, nh = int(vw * scale), int(vh * scale)
    resized = cv2.resize(det_frame, (nw, nh))
    ox = (VIDEO_W - nw) // 2
    oy = TITLE_H + (VIDEO_H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    sx, sy = nw / vw, nh / vh

    def _mx(x):
        return ox + int(x * sx)

    def _my(y):
        return oy + int(y * sy)

    # Rider boxes, bike boxes, head regions and verdicts.
    # Tags are drawn in two passes -- decided verdicts
    # (HELMET / NO HELMET) first, the rest after -- and a tag
    # is skipped when it would overlap one already drawn: in
    # the demo clip's dense queue the riders stand shoulder to
    # shoulder, and the first render printed a whole row of
    # tags on top of each other into unreadable mush. The boxes
    # and the board still show every rider; only the tag text
    # is thinned.
    drawn_tags = []

    def _tag_overlaps(lx, ly, tw, th):
        for (ax, ay, aw, ah) in drawn_tags:
            if (lx < ax + aw and lx + tw > ax
                    and ly < ay + ah and ly + th > ay):
                return True
        return False

    def _draw_tag(tid, tr):
        x, y, w, h, _ = tr["person"][:5]
        verdict = tr["rider"]["confirmed"]
        colour = (RED if verdict == "NO HELMET" else
                  GREEN if verdict == "HELMET" else GREY)
        tag = f"rider #{tid}"
        if verdict is None:
            tag += "  --"
        else:
            tag += f"  {verdict}"
            if verdict in ("HELMET", "NO HELMET"):
                tag += f" {tr['rider']['score']:.2f}"
        if verdict == "NO HELMET":
            tag += "  NO HELMET -- REVIEW"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.55, 2)
        lx = _mx(x)
        if lx + tw + 8 > W - 4:
            lx = max(4, W - 4 - tw - 8)
        ly = _my(y) - 8
        if ly - th < oy + 4:
            ly = _my(y + h) + th + 10
        if _tag_overlaps(lx, ly - th - 4, tw + 8, th + 8):
            return
        drawn_tags.append((lx, ly - th - 4, tw + 8, th + 8))
        cv2.rectangle(canvas, (lx, ly - th - 4),
                      (lx + tw + 8, ly + 4), (24, 26, 32), -1)
        cv2.putText(canvas, tag, (lx + 4, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    live = [(tid, tr) for tid, tr in tracker.tracks.items()
            if tr["missed"] <= 2]
    for tid, tr in live:
        # A track YOLO has not seen for 3 detection rounds (0.3 s)
        # is stale: do not draw its last box.
        x, y, w, h, _ = tr["person"][:5]
        bx, by, bw, bh, _ = tr["bike"][:5]
        verdict = tr["rider"]["confirmed"]
        colour = (RED if verdict == "NO HELMET" else
                  GREEN if verdict == "HELMET" else GREY)
        cv2.rectangle(canvas, (_mx(bx), _my(by)),
                      (_mx(bx + bw), _my(by + bh)), GREY, 1)
        cv2.rectangle(canvas, (_mx(x), _my(y)), (_mx(x + w), _my(y + h)),
                      colour, 3 if verdict == "NO HELMET" else 2)
        # Head region.
        rect = head_rect(tr["person"])
        if rect is not None:
            hx0, hy0, hx1, hy1 = rect
            cv2.rectangle(canvas, (_mx(hx0), _my(hy0)),
                          (_mx(hx1), _my(hy1)), YELLOW, 1)
    # Decided verdicts first so they win the space.
    for tid, tr in live:
        if tr["rider"]["confirmed"] in ("HELMET", "NO HELMET"):
            _draw_tag(tid, tr)
    for tid, tr in live:
        if tr["rider"]["confirmed"] not in ("HELMET", "NO HELMET"):
            _draw_tag(tid, tr)

    # Bottom panel: board + alert banner.
    py = TITLE_H + VIDEO_H
    cv2.rectangle(canvas, (0, py), (W, H), PANEL_BG, -1)
    alert = res["alert"]
    acol = RED if res["any_no_helmet"] else GREEN
    cv2.rectangle(canvas, (0, py), (W, py + 6), acol, -1)
    banner = f"HELMET: {alert}"
    if res["any_no_helmet"]:
        banner += "  --  HUMAN REVIEW ONLY, NOT A FINE"
    cv2.putText(canvas, banner, (16, py + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, acol, 2)
    decided = tracker.helmet + tracker.no_helmet
    comp = (f"{100.0 * tracker.helmet / decided:.0f}%"
            if decided else "--")
    cv2.putText(canvas, f"t = {elapsed:.1f} s", (16, py + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas,
                f"compliance: {comp}   HELMET: {tracker.helmet}   "
                f"NO HELMET: {tracker.no_helmet}   "
                f"UNCLEAR: {tracker.unclear}",
                (300, py + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas,
                f"riders: {tracker.riders_seen}   "
                f"assessed: "
                f"{tracker.helmet + tracker.no_helmet + tracker.unclear}   "
                f"NO HELMET events: {tracker.no_helmet_events}   "
                f"persons: {tracker.person_detections}   "
                f"bikes: {tracker.motorcycle_detections}",
                (16, py + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)
    return canvas


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        root.destroy()
        return path
    except Exception:
        return ""


def open_source(source):
    if source == "pick":
        path = pick_file()
        if not path:
            print("No file chosen.")
            sys.exit(0)
        source = path
    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        sys.exit(1)
    return cap


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source", default="pick",
                    help="video file, webcam index, or 'pick'")
    ap.add_argument("--simulate", action="store_true",
                    help="rendered roadside view instead of a camera/video")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--loop", action="store_true",
                    help="loop a short source until --max-seconds is reached")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--weights", default=None,
                    help="path to yolov8n.onnx (default: next to the script)")
    args = ap.parse_args()

    fps = 30
    here = os.path.dirname(os.path.abspath(__file__))
    detector = None if args.simulate else YoloRiderDetector(
        args.weights or os.path.join(here, "yolov8n.onnx"))
    tracker = HelmetTracker()
    sim = RoadsideSim(DET_W, DET_H, fps) if args.simulate else None
    cap = None if args.simulate else open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else fps
    if not src_fps or src_fps <= 0:
        src_fps = fps
    step = max(1, round(src_fps / fps))  # sample the source at ~30 fps

    out_path = os.path.abspath(args.out)
    base, _ = os.path.splitext(out_path)
    events_path = base + "_events.log"
    summary_path = base + "_summary.txt"
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    mode_label = "SIMULATION (logic check)" if args.simulate else "REAL MODE"

    n_out = 0
    max_frames = int(args.max_seconds * fps)
    src_idx = 0
    res = {"alert": "MONITORING", "any_no_helmet": False,
           "compliance": None}
    while n_out < max_frames:
        t = n_out / fps
        if args.simulate:
            raw = sim.frame()
            det_frame = raw
            det = sim_rider_boxes(det_frame)
            pairs = associate_riders(
                [p + [0] for p in det["persons"]],
                [b + [0] for b in det["motorcycles"]])
            tracker.person_detections += len(det["persons"])
            tracker.motorcycle_detections += len(det["motorcycles"])
            res = tracker.update(pairs, det_frame, t)
        else:
            ok, frame = cap.read()
            src_idx += 1
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    src_idx = 1
                    tracker.reset(t)
                if not ok:
                    break
            if (src_idx - 1) % step != 0:
                continue
            det_frame = cv2.resize(frame, (DET_W, DET_H))
            if n_out % DETECT_EVERY == 0:
                det = detector.detect(det_frame)
                pairs = associate_riders(det["persons"],
                                         det["motorcycles"])
                tracker.person_detections += len(det["persons"])
                tracker.motorcycle_detections += len(det["motorcycles"])
                res = tracker.update(pairs, det_frame, t)
            else:
                res = {"alert": res["alert"],
                       "any_no_helmet": any(
                           tr["rider"]["confirmed"] == "NO HELMET"
                           for tr in tracker.tracks.values()),
                       "compliance": res["compliance"]}
        canvas = draw_canvas(det_frame, tracker, res, t, mode_label)
        writer.write(canvas)
        n_out += 1
        if not args.no_display:
            try:
                cv2.imshow("Helmet Detection", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            except cv2.error:
                pass

    writer.release()
    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    tracker.video_frames = n_out
    with open(events_path, "w") as f:
        for t, text in tracker.events:
            f.write(f"{t:6.1f}s  {text}\n")
    with open(summary_path, "w") as f:
        f.write(tracker.summary())
    print(tracker.summary())
    print(f"video:   {out_path}")
    print(f"events:  {events_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
