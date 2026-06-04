#!/usr/bin/env python3
# run_pipeline_orangepi.py
# 2-stage pipeline: label detector (960) -> defects detector (640)
# Writes:
#   - frame-level metadata.csv (kept from previous behavior)
#   - label-level label_metadata.csv (new tracking aggregation)
#   - video with label boxes
#   - original crops (no boxes)
#   - inference crops (defect boxes)

import os
import csv
import time
from dataclasses import dataclass

import cv2
import numpy as np

from rknn_models import RKNNDetector
from utils_draw import draw_boxes_xyxy

try:
    from lap import lapjv as _lapjv
    HAS_LAP = True
except Exception:
    try:
        from lapx import lapjv as _lapjv
        HAS_LAP = True
    except Exception:
        _lapjv = None
        HAS_LAP = False

# ===================== PATHS =====================
LABEL_MODEL_PATH = "label_detector_fp16.rknn"
DEFECT_MODEL_PATH = "label_defects_detector_fp16.rknn"

PRINTED_DEF_SHAPE = False

SOURCE_PATH = "vid_1.mp4"  # "video.mp4" / "image.jpg" / "0"

OUT_DIR = "out_pipeline_vid_1.4"
METADATA_FILE = os.path.join(OUT_DIR, "metadata.csv")
TRACK_METADATA_FILE = os.path.join(OUT_DIR, "label_metadata.csv")

OUT_VIDEO_LABEL = os.path.join(OUT_DIR, "label_det_video.mp4")

DIR_ORIG = os.path.join(OUT_DIR, "original_crops")
DIR_INF = os.path.join(OUT_DIR, "inference_crops")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DIR_ORIG, exist_ok=True)
os.makedirs(DIR_INF, exist_ok=True)

# ===================== PIPELINE PARAMS =====================
LABEL_INPUT_SIZE = 960
LABEL_CONF_TH = 0.4
LABEL_IOU_TH = 0.45

DEF_INPUT_SIZE = 640
DEF_CONF_TH = 0.6
DEF_IOU_TH = 0.45

CONF_THR_LABEL = 0.25
MIN_AREA = 0.20
MARGIN = 10

BOTTOM_RATIO = 0.25
SHARPNESS_THR = 0
CONTRAST_THR = 0
ENHANCE = False

CONF_THRESHOLDS_DEFECT = {
    "anchor_icon": 0.6,
    "anchor_lot": 0.6,
    "anchor_volume": 0.6,
    "target_field": 0.6,
}
DEFECT_CLASS_NAMES = ["anchor_lot", "anchor_icon", "anchor_volume", "target_field"]

DEFECT_COLORS = {
    0: (0, 255, 255),   # anchor_lot
    1: (0, 255, 0),     # anchor_icon
    2: (0, 165, 255),   # anchor_volume
    3: (255, 0, 0),     # target_field
}

# Tracker params
TRACK_IOU_MATCH = 0.25
TRACK_MAX_MISSED = 12


# ===================== HELPERS =====================
def is_video_file(path: str) -> bool:
    p = path.lower()
    return p.endswith((".mp4", ".avi", ".mov", ".mkv", ".m4v"))


def is_image_file(path: str) -> bool:
    p = path.lower()
    return p.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def gray_contrast(gray: np.ndarray) -> float:
    return float(np.std(gray))


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def area_ratio(x1, y1, x2, y2, w, h) -> float:
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    return (bw * bh) / float(w * h + 1e-9)


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def pick_best_label_idx(dets):
    if not dets:
        return None
    best_i = 0
    best_s = float(dets[0][4])
    for i in range(1, len(dets)):
        s = float(dets[i][4])
        if s > best_s:
            best_s = s
            best_i = i
    return best_i


def is_valid_label_det(det, frame_w: int, frame_h: int) -> bool:
    x1, y1, x2, y2, conf, _ = det
    if float(conf) < float(CONF_THR_LABEL):
        return False
    ar = area_ratio(int(x1), int(y1), int(x2), int(y2), frame_w, frame_h)
    if ar < float(MIN_AREA):
        return False
    if x1 < MARGIN or y1 < MARGIN or x2 > frame_w - MARGIN or y2 > frame_h - MARGIN:
        return False
    return True


def ensure_frame_csv_header(path: str):
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow([
            "id", "source", "frame",
            "x1", "y1", "x2", "y2", "label_conf",
            "sharpness", "contrast",
            "orig_filename", "infer_filename",
            "is_defective",
            "anchor_icon", "anchor_lot", "anchor_volume", "target_field_count",
        ])
    return f, w


def ensure_track_csv_header(path: str):
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow([
            "track_id", "source",
            "start_frame", "end_frame", "frames_analyzed",
            "ok_frames", "defective_frames", "defective_rate",
            "label_conf_avg", "sharpness_avg", "contrast_avg",
            "anchor_icon_sum", "anchor_lot_sum", "anchor_volume_sum", "target_field_sum",
            "is_defective_any",
        ])
    return f, w


def counts_from_defects(dets_def):
    counts = {name: 0 for name in DEFECT_CLASS_NAMES}
    for (_, _, _, _, s, cid) in dets_def:
        if cid < 0 or cid >= len(DEFECT_CLASS_NAMES):
            continue
        name = DEFECT_CLASS_NAMES[cid]
        thr = CONF_THRESHOLDS_DEFECT.get(name, 0.7)
        if float(s) >= float(thr):
            counts[name] += 1
    return counts


def decide_defective(counts):
    has_icon = counts.get("anchor_icon", 0) == 1
    has_lot = counts.get("anchor_lot", 0) == 1
    has_target = counts.get("target_field", 0) == 2
    return not (has_icon and has_lot and has_target)


def progress_line(frames_done, frames_total, t_start, saved_total):
    elapsed = time.time() - t_start
    fps = frames_done / elapsed if elapsed > 0 else 0.0

    if frames_total and frames_total > 0:
        pct = 100.0 * frames_done / frames_total
        remaining = frames_total - frames_done
        eta = remaining / fps if fps > 0 else 0.0

        bar_len = 24
        filled = int(bar_len * frames_done / frames_total)
        bar = "#" * filled + "." * (bar_len - filled)
        return f"\r[{bar}] {pct:6.2f}%  frames={frames_done}/{frames_total}  saved={saved_total}  fps={fps:5.2f}  eta={eta:6.1f}s"
    return f"\rframes={frames_done}  saved={saved_total}  fps={fps:5.2f}  elapsed={elapsed:6.1f}s"


@dataclass
class TrackState:
    track_id: int
    bbox: tuple
    last_frame: int
    missed: int = 0
    hits: int = 1


class LabelTracker:
    def __init__(self, iou_match=0.25, max_missed=10):
        self.iou_match = float(iou_match)
        self.max_missed = int(max_missed)
        self.next_track_id = 1
        self.tracks = {}  # track_id -> TrackState

    def _assign_lap(self, track_ids, det_boxes):
        n_t = len(track_ids)
        n_d = len(det_boxes)
        det_to_track = {}
        if n_t == 0 or n_d == 0:
            return det_to_track

        cost = np.full((n_t, n_d), 1e6, dtype=np.float32)
        for i, tid in enumerate(track_ids):
            tbox = self.tracks[tid].bbox
            for j, dbox in enumerate(det_boxes):
                iou = iou_xyxy(tbox, dbox)
                if iou >= self.iou_match:
                    cost[i, j] = 1.0 - iou

        if HAS_LAP:
            try:
                _, x, _ = _lapjv(cost, extend_cost=True, cost_limit=(1.0 - self.iou_match))
            except TypeError:
                _, x, _ = _lapjv(cost)
            for i, j in enumerate(x):
                if j < 0 or j >= n_d:
                    continue
                if cost[i, j] < 1e5:
                    det_to_track[j] = track_ids[i]
            return det_to_track

        # Greedy fallback if LAP package is not available.
        used_tracks = set()
        used_dets = set()
        while True:
            best_i = -1
            best_j = -1
            best_cost = 1e9
            for i in range(n_t):
                if i in used_tracks:
                    continue
                for j in range(n_d):
                    if j in used_dets:
                        continue
                    c = cost[i, j]
                    if c < best_cost:
                        best_cost = c
                        best_i = i
                        best_j = j
            if best_i < 0 or best_j < 0 or best_cost >= 1e5:
                break
            used_tracks.add(best_i)
            used_dets.add(best_j)
            det_to_track[best_j] = track_ids[best_i]
        return det_to_track

    def update(self, dets, frame_id):
        det_boxes = [(int(d[0]), int(d[1]), int(d[2]), int(d[3])) for d in dets]
        track_ids = list(self.tracks.keys())
        det_to_track = self._assign_lap(track_ids, det_boxes)

        matched_tracks = set()
        for det_idx, tid in det_to_track.items():
            matched_tracks.add(tid)
            st = self.tracks[tid]
            st.bbox = det_boxes[det_idx]
            st.last_frame = frame_id
            st.missed = 0
            st.hits += 1

        for tid in list(self.tracks.keys()):
            if tid in matched_tracks:
                continue
            st = self.tracks[tid]
            st.missed += 1
            if st.missed > self.max_missed:
                del self.tracks[tid]

        for det_idx, box in enumerate(det_boxes):
            if det_idx in det_to_track:
                continue
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracks[tid] = TrackState(track_id=tid, bbox=box, last_frame=frame_id)
            det_to_track[det_idx] = tid

        return det_to_track


class TrackAggregator:
    def __init__(self):
        self.stats = {}

    def update(self, source_name: str, track_id: int, frame_id: int, row: dict):
        st = self.stats.get(track_id)
        if st is None:
            st = {
                "track_id": track_id,
                "source": source_name,
                "start_frame": frame_id,
                "end_frame": frame_id,
                "frames_analyzed": 0,
                "defective_frames": 0,
                "ok_frames": 0,
                "sum_conf": 0.0,
                "sum_sharp": 0.0,
                "sum_contrast": 0.0,
                "anchor_icon_sum": 0,
                "anchor_lot_sum": 0,
                "anchor_volume_sum": 0,
                "target_field_sum": 0,
            }
            self.stats[track_id] = st

        st["end_frame"] = frame_id
        st["frames_analyzed"] += 1
        st["defective_frames"] += int(row["is_defective"])
        st["ok_frames"] += int(not row["is_defective"])
        st["sum_conf"] += float(row["label_conf"])
        st["sum_sharp"] += float(row["sharpness"])
        st["sum_contrast"] += float(row["contrast"])
        st["anchor_icon_sum"] += int(row["anchor_icon"])
        st["anchor_lot_sum"] += int(row["anchor_lot"])
        st["anchor_volume_sum"] += int(row["anchor_volume"])
        st["target_field_sum"] += int(row["target_field_count"])

    def write_csv(self, writer):
        for track_id in sorted(self.stats.keys()):
            st = self.stats[track_id]
            n = max(1, int(st["frames_analyzed"]))
            defect_rate = st["defective_frames"] / float(n)
            writer.writerow([
                st["track_id"], st["source"],
                st["start_frame"], st["end_frame"], st["frames_analyzed"],
                st["ok_frames"], st["defective_frames"], round(defect_rate, 4),
                round(st["sum_conf"] / n, 4),
                round(st["sum_sharp"] / n, 2),
                round(st["sum_contrast"] / n, 2),
                st["anchor_icon_sum"], st["anchor_lot_sum"], st["anchor_volume_sum"], st["target_field_sum"],
                int(defect_rate >= 0.5),
            ])


def analyze_label_roi(
    frame_bgr,
    frame_id,
    source_name,
    defect_det: RKNNDetector,
    global_id,
    frame_csv_writer,
    track_id: int,
    label_det_tuple,
):
    x1, y1, x2, y2, conf, _ = label_det_tuple
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return global_id, 0, None

    ch, cw = crop.shape[:2]
    y_cut = int(ch * (1 - float(BOTTOM_RATIO)))
    crop_bottom = crop[y_cut:ch, 0:cw]
    if crop_bottom.size == 0:
        return global_id, 0, None

    gray = cv2.cvtColor(crop_bottom, cv2.COLOR_BGR2GRAY)
    sharp = laplacian_sharpness(gray)
    contr = gray_contrast(gray)
    if sharp < float(SHARPNESS_THR) or contr < float(CONTRAST_THR):
        return global_id, 0, None

    if ENHANCE:
        enh = enhance_gray(gray)
        crop_bottom_in = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
    else:
        crop_bottom_in = crop_bottom

    def_dets, _, out2_shape = defect_det.infer(crop_bottom_in)

    global PRINTED_DEF_SHAPE
    if not PRINTED_DEF_SHAPE:
        print("DEFECT out shape:", out2_shape)
        PRINTED_DEF_SHAPE = True

    counts = counts_from_defects(def_dets)
    is_def = decide_defective(counts)

    det_idx = 0
    fname_base = f"frame_{frame_id:06d}_det_{det_idx:02d}"
    fname_orig = f"{fname_base}.jpg"
    out_path_orig = os.path.join(DIR_ORIG, fname_orig)
    if not cv2.imwrite(out_path_orig, crop_bottom_in):
        raise RuntimeError(f"cv2.imwrite failed: {out_path_orig}")

    crop_inf = draw_boxes_xyxy(
        crop_bottom_in,
        def_dets,
        class_names=DEFECT_CLASS_NAMES,
        colors=DEFECT_COLORS,
        box_thick=2,
        text_scale=0.6,
        text_thick=2,
    )
    fname_inf = f"{fname_base}_inf.jpg"
    out_path_inf = os.path.join(DIR_INF, fname_inf)
    if not cv2.imwrite(out_path_inf, crop_inf):
        raise RuntimeError(f"cv2.imwrite failed: {out_path_inf}")

    row = {
        "id": global_id,
        "source": source_name,
        "frame": frame_id,
        "track_id": int(track_id),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "label_conf": round(float(conf), 4),
        "sharpness": round(float(sharp), 2),
        "contrast": round(float(contr), 2),
        "orig_filename": fname_orig,
        "infer_filename": fname_inf,
        "is_defective": int(is_def),
        "anchor_icon": counts.get("anchor_icon", 0),
        "anchor_lot": counts.get("anchor_lot", 0),
        "anchor_volume": counts.get("anchor_volume", 0),
        "target_field_count": counts.get("target_field", 0),
    }

    frame_csv_writer.writerow([
        row["id"], row["source"], row["frame"],
        row["x1"], row["y1"], row["x2"], row["y2"], row["label_conf"],
        row["sharpness"], row["contrast"],
        row["orig_filename"], row["infer_filename"],
        row["is_defective"],
        row["anchor_icon"], row["anchor_lot"], row["anchor_volume"], row["target_field_count"],
    ])

    return global_id + 1, 1, row


def process_one_frame(
    frame_bgr,
    frame_id,
    source_name,
    label_det: RKNNDetector,
    defect_det: RKNNDetector,
    tracker: LabelTracker,
    aggregator: TrackAggregator,
    global_id,
    frame_csv_writer,
):
    h, w = frame_bgr.shape[:2]
    label_dets, _, _ = label_det.infer(frame_bgr)

    frame_label_vis = draw_boxes_xyxy(
        frame_bgr,
        label_dets,
        class_names=["label"],
        colors={0: (0, 255, 255)},
        box_thick=2,
        text_scale=0.7,
        text_thick=2,
    )

    valid_dets = [d for d in label_dets if is_valid_label_det(d, w, h)]
    det_to_track = tracker.update(valid_dets, frame_id)
    best_idx = pick_best_label_idx(valid_dets)

    # Draw track ids for every valid tracked label to simplify video review.
    for det_idx, det in enumerate(valid_dets):
        track_id_i = det_to_track.get(det_idx)
        if track_id_i is None:
            continue
        x1_i, y1_i, _, _, _, _ = det
        cv2.putText(
            frame_label_vis,
            f"track:{track_id_i}",
            (int(x1_i), max(0, int(y1_i) - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if best_idx is None:
        return global_id, 0, frame_label_vis

    best_det = valid_dets[best_idx]
    track_id = det_to_track.get(best_idx)
    if track_id is None:
        return global_id, 0, frame_label_vis

    # Mark the selected box used for defect analysis in this frame.
    x1, y1, _, _, _, _ = best_det
    cv2.putText(
        frame_label_vis,
        f"*track:{track_id}",
        (int(x1), max(0, int(y1) - 36)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )

    global_id_next, saved, row = analyze_label_roi(
        frame_bgr=frame_bgr,
        frame_id=frame_id,
        source_name=source_name,
        defect_det=defect_det,
        global_id=global_id,
        frame_csv_writer=frame_csv_writer,
        track_id=track_id,
        label_det_tuple=best_det,
    )
    if row is not None:
        aggregator.update(source_name=source_name, track_id=track_id, frame_id=frame_id, row=row)
    return global_id_next, saved, frame_label_vis


# ===================== MAIN =====================
def main():
    frame_csv_file, frame_csv_writer = ensure_frame_csv_header(METADATA_FILE)
    track_csv_file, track_csv_writer = ensure_track_csv_header(TRACK_METADATA_FILE)
    global_id = 1

    tracker = LabelTracker(iou_match=TRACK_IOU_MATCH, max_missed=TRACK_MAX_MISSED)
    aggregator = TrackAggregator()
    if not HAS_LAP:
        print("Tracker note: lap/lapx is not available, using greedy assignment fallback.")

    label_det = RKNNDetector(
        LABEL_MODEL_PATH,
        LABEL_INPUT_SIZE,
        LABEL_CONF_TH,
        LABEL_IOU_TH,
        num_classes=1,
        has_obj=False,
        debug_first_infer=True,
    )
    defect_det = RKNNDetector(
        DEFECT_MODEL_PATH,
        DEF_INPUT_SIZE,
        DEF_CONF_TH,
        DEF_IOU_TH,
        num_classes=4,
        has_obj=False,
        debug_first_infer=True,
    )

    label_det.open()
    defect_det.open()

    try:
        if is_image_file(SOURCE_PATH):
            img = cv2.imread(SOURCE_PATH)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {SOURCE_PATH}")

            global_id, saved, frame_label_vis = process_one_frame(
                frame_bgr=img,
                frame_id=1,
                source_name=os.path.basename(SOURCE_PATH),
                label_det=label_det,
                defect_det=defect_det,
                tracker=tracker,
                aggregator=aggregator,
                global_id=global_id,
                frame_csv_writer=frame_csv_writer,
            )
            frame_csv_file.flush()
            aggregator.write_csv(track_csv_writer)
            track_csv_file.flush()

            out_img_label = os.path.join(OUT_DIR, "label_det_image.jpg")
            cv2.imwrite(out_img_label, frame_label_vis)

            print(f"Done image. Saved crops: {saved}.")
            print(f"Frame CSV: {os.path.abspath(METADATA_FILE)}")
            print(f"Track CSV: {os.path.abspath(TRACK_METADATA_FILE)}")
            print(f"Label vis image: {os.path.abspath(out_img_label)}")
            return

        if is_video_file(SOURCE_PATH):
            cap = cv2.VideoCapture(SOURCE_PATH)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {SOURCE_PATH}")
            source_name = os.path.basename(SOURCE_PATH)
            frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif SOURCE_PATH.isdigit():
            cap = cv2.VideoCapture(int(SOURCE_PATH))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera: {SOURCE_PATH}")
            source_name = f"cam_{SOURCE_PATH}"
            frames_total = 0
        else:
            raise RuntimeError("SOURCE_PATH must be image/video or camera index string like '0'.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or fps != fps:
            fps = 25.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            ok, tmp = cap.read()
            if not ok:
                raise RuntimeError("Cannot read first frame to get size.")
            h, w = tmp.shape[:2]
            if is_video_file(SOURCE_PATH):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_video = cv2.VideoWriter(OUT_VIDEO_LABEL, fourcc, fps, (w, h))
        if not writer_video.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter: {OUT_VIDEO_LABEL}")
        print(f"Output dir: {os.path.abspath(OUT_DIR)}")
        print(f"Saving label video to: {os.path.abspath(OUT_VIDEO_LABEL)}")
        print("Stop with Ctrl+C.\n")

        frame_id = 0
        saved_total = 0
        t0 = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1

                global_id, saved, frame_label_vis = process_one_frame(
                    frame_bgr=frame,
                    frame_id=frame_id,
                    source_name=source_name,
                    label_det=label_det,
                    defect_det=defect_det,
                    tracker=tracker,
                    aggregator=aggregator,
                    global_id=global_id,
                    frame_csv_writer=frame_csv_writer,
                )
                saved_total += saved
                writer_video.write(frame_label_vis)

                if frame_id % 30 == 0:
                    frame_csv_file.flush()
                if frame_id % 5 == 0:
                    print(progress_line(frame_id, frames_total, t0, saved_total), end="", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Finishing...")

        t1 = time.time()
        cap.release()
        writer_video.release()

        aggregator.write_csv(track_csv_writer)
        frame_csv_file.flush()
        track_csv_file.flush()

        total_time = t1 - t0
        avg_fps = frame_id / total_time if total_time > 0 else 0.0
        avg_ms = (total_time / frame_id) * 1000.0 if frame_id > 0 else 0.0

        print("\n\n========== PIPELINE RESULT ==========")
        print(f"Frames read: {frame_id}")
        print(f"Crops saved: {saved_total}")
        print(f"Total time: {total_time:.3f} sec")
        print(f"Average time per frame: {avg_ms:.2f} ms")
        print(f"Average pipeline FPS: {avg_fps:.2f}")
        print(f"Output dir: {os.path.abspath(OUT_DIR)}")
        print(f"Frame CSV: {os.path.abspath(METADATA_FILE)}")
        print(f"Track CSV: {os.path.abspath(TRACK_METADATA_FILE)}")
        print("=====================================\n")

    finally:
        label_det.close()
        defect_det.close()
        frame_csv_file.close()
        track_csv_file.close()


if __name__ == "__main__":
    main()
