# Helmet Detection — motorcycle rider compliance

A roadside / traffic-camera helmet check for motorcycle riders. YOLOv8n
finds motorcycles and riders (persons), each person is associated with a
motorcycle (overlap / sitting above the bike box), and the head region of
each rider — the top ~25% of the person box — is checked for a helmet,
giving a **HELMET / NO HELMET / UNCLEAR** verdict per rider. A NO HELMET
rider raises a red banner and a timestamped event-log entry, and the
per-rider board shows riders tracked, helmets, no-helmets, unclear, and
the compliance rate among decided riders.

**A helmet verdict is a flag FOR HUMAN REVIEW, never a fine.** This is
stated on the banner, in the event log, and here. A no-helmet rider can
be linked to the bike's plate via the license-plate-recognition code in
this portfolio. The WHO notes helmets cut rider death risk by more than
six times; several 2026 papers pair YOLOv8 helmet detection with ANPR
for exactly this enforcement-assist role.

## How a rider is judged

* **Rider association** — a person box counts as a rider only when it
  overlaps a motorcycle box or sits above it. Unpaired persons
  (pedestrians on the bridge, bystanders) are not judged.
* **Head region** — the top quarter of the rider's person box, inset
  15% on the sides. It is a documented image region for this scene,
  not a calibrated camera model: on a rider the head sits at the top
  of the person box. A pillion rider's head is smaller and partly
  hidden behind the driver, so it reads UNCLEAR more often.
* **Helmet check** — a helmet is a smooth rounded shell: one solid
  colour covering the whole top of the head. A bare head is hair on
  top and face below. Two measurements on the head crop:
  1. **Shell blob** — the crop is blurred (5×5) and quantised to
     coarse colour bins; the largest connected blob of the dominant
     colour is measured in the **central band** of the crop (the
     middle 60% of its width, where the head sits). The full crop's
     dominant blob is often the background — dark trees beside the
     head read as a dark "shell" and caused false NO HELMETs in an
     earlier build. A shell is a big bright blob at the top of the
     band (blob ≥ 0.30, mean V ≥ 65).
  2. **Skin fraction** — HSV skin range (H 0–30, S 30–180, V 40–255;
     S is capped at 180 because a saturated red helmet sits in the
     skin hue band). A bare face fills the crop with skin; a
     helmeted head shows only a sliver of face.
* **Verdicts** — HELMET needs a bright top-cover blob ≥ 0.30;
  NO HELMET needs skin fraction ≥ 0.45; anything else, or a head
  too small (< 50 px person box), too dark (mean brightness < 40),
  or clipped by the frame edge, reads UNCLEAR. A dark top blob
  (hair, or a dark helmet, or dark background) reads UNCLEAR —
  never NO HELMET. Only a clearly bare face, skin filling nearly
  half the crop, raises the flag.
* **Confirmation** — a verdict must repeat in 3 consecutive
  detection rounds before it counts and is logged, so one
  flickering YOLO box or one glinting visor cannot raise a flag.
  The tag next to a decided rider shows the live shell-blob
  fraction (e.g. `HELMET 0.32`) as a diagnostic.

### Calibration (measured, not assumed)

* Simulation (rendered riders, same pixel pipeline): helmeted
  shells blob 0.55–0.62, V 220–234, skin 0.05; bare heads blob
  0.34, V 210, skin 0.45.
* Real demo clip, sampled head crops: helmeted heads blob
  0.09–0.60, skin 0.00–0.13; the highest helmeted-face skin
  reading was 0.42 (one close red-helmet rider) — the NO HELMET
  bar of 0.45 sits above it.
* Edge density was measured and rejected as a signal: on the demo
  clip every helmeted head read edge 0.11–0.26, the same range as
  a bare head (visor edges, vents, neighbouring riders, clutter).
  It is logged only as a diagnostic.

## Run it

```bash
pip install -r requirements.txt
python helmet_detection.py                 # file picker (tkinter)
python helmet_detection.py --source video.mp4
python helmet_detection.py --source 0      # webcam
python helmet_detection.py --simulate      # simulation mode
```

Options: `--loop` (loop a short clip; the loop point is logged and
tracks reset), `--max-seconds N`, `--no-display` (headless), `--out
path.mp4`. Each run writes `output.mp4`, `output_events.log` and
`output_summary.txt` next to the output path (default names are
`output.mp4` etc. in the current directory). Press **Q** to quit.

Runtime deps are only `opencv-python-headless` and `numpy` — YOLOv8n
runs through OpenCV DNN, not ultralytics. The ONNX output is
`[1, 84, N]` = 4 box coords + 80 class scores, with **no objectness
channel**: confidence is the max class score and the class is its
argmax. `yolov8n.onnx` ships in this repo (12.8 MB); if it is missing
it is downloaded from the Ultralytics assets release.

## Demo — real footage (what the demo video shows)

30 s, 1280×720 @ 30 fps, real mode over a fresh Pexels roadside clip
of a queue of scooters/motorcycles at a junction (Pexels video
36306512; the 19.0 s clip looped once at 19.0 s, loop logged, tracks
reset). The clip contains only helmeted riders — **no bare-headed
rider appears in it**, so the demo shows HELMET confirmations and
UNCLEAR reads, and zero NO HELMET flags. The NO HELMET path is
verified only in simulation (below).

End-of-run counters (final verified run, 25 Nov 2026):

| measure | value |
|---|---|
| frames processed | 900 |
| tracker updates (YOLO rounds) | 300 |
| person detections (summed) | 1,967 |
| motorcycle detections (summed) | 978 |
| riders tracked | 67 |
| riders assessed (confirmed verdicts) | 57 |
| HELMET | 13 |
| NO HELMET (flagged for human review) | 0 |
| UNCLEAR | 44 |
| compliance among decided riders | 100.0% (13/13) |
| NO HELMET events logged | 0 |
| events logged | 24 |
| head crops HELMET | shell blob 0.32–0.54, skin 0.00–0.11 |

The demo video is `../project-demos/helmet-detection-demo.mp4`
(H.264 copy in `../project-demos/linkedin-h264/`), with its event log
and summary alongside it. Visually inspected frames at 5 s, 12 s and
25 s: rider boxes, yellow head rects, bike boxes and the compliance
board all render; the 13 HELMET confirmations are the clearest
shells (white/red/light-blue helmets, blob 0.32–0.54).

## Simulation — internal logic check only (never the demo)

30 s, `--simulate`: three rendered bikes cross the frame — bike 1 a
red-helmet driver (0–12 s), bike 2 a bare driver (4–18 s), bike 3 a
white-helmet driver with a bare pillion (16–28 s). Judged through
the same pixel pipeline (colour-mask person/bike detection instead
of YOLO; the verdict code is identical).

| measure | value |
|---|---|
| riders tracked | 4 |
| HELMET | 2 |
| NO HELMET (flagged for human review) | 2 |
| UNCLEAR | 0 |
| compliance among decided riders | 50.0% (2/4) |
| NO HELMET events logged | 2 |
| first NO HELMET at | 5.3 s |

Event log reads exactly as scripted: rider #1 HELMET at 1.2 s, rider
#2 NO HELMET at 5.3 s, rider #3 HELMET at 17.2 s, rider #4 (pillion)
NO HELMET at 17.9 s, with UNCLEAR transitions as each rider leaves
the frame. All 12 sampled sim verdicts were correct.

## Bugs found and fixed during verification

1. **Sim skin tone outside its own mask** — the first skin tone
   (S ~58) sat below the mask floor (S 60), so the simulation found
   zero riders. Skin tone changed to BGR (90, 140, 210), S ~146.
2. **Sim sky inside the skin range** — the first sky colour (hue 13,
   S 70) was detected as one giant 1280 px "person". Sky changed to
   blue, then that sky (S ~127) sat inside the blue motorcycle mask
   and read as a giant "motorcycle". Sky is now pale grey-blue
   BGR (190, 170, 150), S ~54.
3. **Sim riders too small for the head gate** — person boxes 54 px
   wide, gate was 60 px; the first sim run judged zero riders.
   Bikes scaled up (240×100, rider height 130) and the gate lowered
   to 50 px.
4. **Pillion merged with driver** in the tan skin mask (52 px
   apart) — pillion moved to 90 px behind the driver.
5. **Person-box growth by fixed %** overshot bare riders (head crop
   filled with uniform road → false HELMET blob) and undershot
   helmeted riders (dome cut in half → false NO HELMET). Replaced
   by a union mask (tan skin + saturated red helmet + dark
   hair/visor/wheels + bright white restricted to the sky band),
   whose bounding box covers the whole rider including the head;
   wheels are filtered out by a box-top floor.
6. **Sim arm widened the person box** (reached 26 px past the
   torso), so the head crop was off-centre and filled with road —
   false HELMET blob. The arm now reaches only 8 px past the torso.
7. **Skin mask S uncapped caught red helmets** (S ~208) as skin —
   S capped at 180.
8. **Head-rect clip check was on the head rect only** — a rider half
   in frame still passed. The check is now on the person box.
9. **Entry-transient false verdicts** (NO HELMET at 1.1 s, HELMET
   from uniform sky at 5.2 s) — addressed by fixes 5 and 8;
   verified gone in later sim runs.
10. **Dark top-blob NO HELMET path** — misread dark trees and dark
    helmets as hair (8 inspected false NO HELMETs on real footage
    were all helmeted riders). The path was dropped: a dark top
    blob now reads UNCLEAR, never NO HELMET.
11. **Full-crop dominant blob was background** — the shell blob is
    now measured in the central band of the crop only, where the
    head sits.
12. **Event log printed the wrong rider id** — the confirmation log
    used `tid`, which is only assigned when a NEW track is created
    and otherwise keeps the last-created track's id, so later
    transitions were logged under the wrong rider (rider #1's exit
    logged as #2, rider #3's as #4). The log now uses the matched
    track id. Counts were never affected (they are keyed by the
    matched id) — only the log text was wrong.
13. **Overlapping rider tags in the dense queue** — the first demo
    render printed a whole row of tags on top of each other into
    unreadable mush. Tags are now drawn decided-verdicts-first and
    skipped when they would overlap one already drawn; boxes and
    the board still show every rider.
14. **Title clipped at the frame edge** — shortened to
    "HELMET DETECTION -- motorcycle riders".

## Honest limits

* **Dark helmets read UNCLEAR by design** — a dark shell is as dark
  as hair, so brightness cannot tell them apart; the check refuses
  to guess. On the demo clip most UNCLEAR reads are dark helmets,
  small/distant heads, or occluded heads in the crowded queue.
* **A close rider whose face fills the crop can read NO HELMET on
  skin alone.** The bar (0.45) sits above the highest helmeted-face
  reading measured on the demo clip (0.42), but a very close,
  unhelmeted-looking face is the known false-positive shape — which
  is why every NO HELMET is a flag for human review, never a fine.
* **Crowded queues fragment tracks.** "Riders tracked" counts track
  segments, not distinct people: in the demo clip 67 tracks cover
  roughly 10–15 distinct riders, because riders occlude each other
  and tracks break and restart. The 13 HELMET confirmations are 13
  track segments; the same rider can confirm HELMET twice (once
  per loop pass — the 19.0 s loop point is logged). Compliance is a
  property of this clip's *decided* riders (13/13), not a claim
  about the road.
* **Pillion riders** are smaller and partly hidden behind the
  driver; in simulation the pillion is judged correctly, but on
  real footage pillions mostly read UNCLEAR.
* **Small/distant heads** (under ~50 px person box, or mean
  brightness under 40) are refused as UNCLEAR rather than guessed.
* **This is a colour/shape heuristic, not a trained helmet
  classifier.** It works on the clear shells in the demo clip and
  in simulation; a production system would train a dedicated
  helmet/no-helmet detector (several 2026 papers do exactly that
  with YOLOv8 + ANPR). The script's job is to show the pipeline —
  detect, associate, judge the head, confirm, log — honestly.
