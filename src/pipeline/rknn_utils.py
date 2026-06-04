# rknn_utils.py
import cv2
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def letterbox_rgb(img_bgr: np.ndarray, new_size: int):
    """
    Ultralytics-like letterbox:
    orig -> resize (keep aspect) -> pad to square -> BGR->RGB
    Returns: rgb(uint8 H,W,3), scale, pad_left, pad_top
    """
    h0, w0 = img_bgr.shape[:2]
    r = min(new_size / w0, new_size / h0)
    new_w = int(round(w0 * r))
    new_h = int(round(h0 * r))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = new_size - new_w
    pad_h = new_size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return rgb, r, pad_left, pad_top

def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_th: float):
    """Classic NMS. boxes: (N,4) xyxy."""
    if boxes.size == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union

        inds = np.where(iou <= iou_th)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)

def _to_CN(out: np.ndarray) -> np.ndarray:
    """Bring to (C,N) float32."""
    a = np.asarray(out).squeeze()
    if a.ndim != 2:
        raise RuntimeError(f"Unexpected output after squeeze: {a.shape}")

    # want (C,N)
    if a.shape[0] <= a.shape[1]:
        # already (C,N)
        pass
    else:
        # (N,C) -> (C,N)
        a = a.T
    return a.astype(np.float32)

def decode_yolo_like_1xCxN(
    out: np.ndarray,
    input_size: int,
    conf_th: float,
    iou_th: float,
    scale: float,
    pad_left: int,
    pad_top: int,
    orig_w: int,
    orig_h: int,
    num_classes: int | None = None,
    has_obj: bool | None = None,
    box_format: str = "xywh",   # "xywh" or "xyxy"
    debug_once: bool = False,
):
    """
    Универсальный декодер для RKNN-выходов (C,N):
      1) C=5                  -> [x,y,w,h,conf] (1 класс)
      2) C=4+nc               -> [x,y,w,h, cls_scores...]  (obj НЕТ)   <-- твой (1,8,8400) при nc=4
      3) C=5+nc               -> [x,y,w,h,obj, cls_scores...]

    Возвращает: list of (x1,y1,x2,y2, score, class_id) в координатах оригинала.
    """
    a = _to_CN(out)
    C, N = a.shape
    if C < 5:
        raise RuntimeError(f"Too few channels: C={C}, shape={a.shape}")

    # boxes
    bx0 = a[0]
    by0 = a[1]
    bw0 = a[2]
    bh0 = a[3]

    # эвристика нормализации
    mx = max(float(np.max(np.abs(bx0))), float(np.max(np.abs(by0))),
             float(np.max(np.abs(bw0))), float(np.max(np.abs(bh0))))
    normalized = mx <= 2.0
    if normalized:
        bx0 = bx0 * input_size
        by0 = by0 * input_size
        bw0 = bw0 * input_size
        bh0 = bh0 * input_size

    # ----- decide layout -----
    if C == 5:
        # single class: conf in a[4]
        conf = a[4]
        if conf.max() > 1.0 or conf.min() < 0.0:
            conf = sigmoid(conf)
        cls_id = np.zeros((N,), dtype=np.int32)
        score = conf
    else:
        # multi-class
        if num_classes is None:
            # guess from C
            # if has_obj is forced -> use it, else guess:
            # prefer "no obj" when C-4 looks like a reasonable nc (>=2)
            if has_obj is None:
                # if C-4 in [2..200] -> likely no obj
                if 2 <= (C - 4) <= 200:
                    has_obj_guess = False
                else:
                    has_obj_guess = True
            else:
                has_obj_guess = has_obj

            if has_obj_guess:
                num_classes = C - 5
            else:
                num_classes = C - 4
        else:
            # num_classes задан — определим has_obj если не задан
            if has_obj is None:
                has_obj = (C == 5 + num_classes)

        if has_obj is None:
            # fallback if still unknown
            has_obj = (C == 5 + num_classes)

        if has_obj:
            if C != 5 + num_classes:
                raise RuntimeError(f"Shape mismatch: C={C} but num_classes={num_classes} expects C={5+num_classes}")
            obj = a[4]
            cls_scores = a[5:5 + num_classes]  # (nc,N)

            if obj.max() > 1.0 or obj.min() < 0.0:
                obj = sigmoid(obj)
            if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
                cls_scores = sigmoid(cls_scores)

            cls_id = np.argmax(cls_scores, axis=0).astype(np.int32)
            cls_max = np.max(cls_scores, axis=0)
            score = obj * cls_max
        else:
            if C != 4 + num_classes:
                raise RuntimeError(f"Shape mismatch: C={C} but num_classes={num_classes} expects C={4+num_classes}")
            cls_scores = a[4:4 + num_classes]  # (nc,N)

            if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
                cls_scores = sigmoid(cls_scores)

            cls_id = np.argmax(cls_scores, axis=0).astype(np.int32)
            score = np.max(cls_scores, axis=0)

    if debug_once and not hasattr(decode_yolo_like_1xCxN, "_dbg_printed"):
        decode_yolo_like_1xCxN._dbg_printed = True
        print(f"[DEBUG] C={C}, N={N}, normalized={normalized}, num_classes={num_classes}, has_obj={has_obj}")
        print("[DEBUG] box sample:", np.stack([bx0, by0, bw0, bh0], axis=1)[:3])
        # покажем пару score
        print("[DEBUG] score sample:", score[:10])
        print("[DEBUG] score min/max:", float(score.min()), float(score.max()))

    # filter by conf
    mask = score >= conf_th
    if not np.any(mask):
        return []

    bx0 = bx0[mask]; by0 = by0[mask]; bw0 = bw0[mask]; bh0 = bh0[mask]
    score = score[mask]
    cls_id = cls_id[mask]

    # boxes to xyxy in letterbox coords
    if box_format == "xywh":
        x1 = bx0 - bw0 / 2.0
        y1 = by0 - bh0 / 2.0
        x2 = bx0 + bw0 / 2.0
        y2 = by0 + bh0 / 2.0
    elif box_format == "xyxy":
        x1, y1, x2, y2 = bx0, by0, bw0, bh0
    else:
        raise ValueError("box_format must be 'xywh' or 'xyxy'")

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS (class-agnostic)
    keep = nms_xyxy(boxes, score, iou_th)
    boxes = boxes[keep]
    score = score[keep]
    cls_id = cls_id[keep]

    # undo letterbox -> orig coords
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= float(scale + 1e-12)

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

    dets = []
    for b, s, c in zip(boxes, score, cls_id):
        x1i, y1i, x2i, y2i = b.astype(int).tolist()
        dets.append((x1i, y1i, x2i, y2i, float(s), int(c)))
    return dets