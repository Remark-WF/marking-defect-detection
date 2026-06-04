# utils_draw.py
import cv2

def draw_boxes_xyxy(
    img_bgr,
    dets,
    class_names=None,
    colors=None,
    box_thick=2,
    text_scale=0.6,
    text_thick=2,
    conf_fmt="{:.2f}",
):
    """
    dets: list of (x1, y1, x2, y2, score, class_id)
    class_names: list[str] or None
    colors: dict[int -> (B,G,R)] or None
    """
    out = img_bgr.copy()
    for (x1, y1, x2, y2, score, cid) in dets:
        if colors and cid in colors:
            color = colors[cid]
        else:
            color = (0, 255, 0)

        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, box_thick)

        if class_names and 0 <= cid < len(class_names):
            name = class_names[cid]
        else:
            name = f"class_{cid}"

        label = f"{name} {conf_fmt.format(float(score))}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thick)

        y0 = max(0, int(y1) - th - 8)
        cv2.rectangle(out, (int(x1), y0), (int(x1) + tw + 6, int(y1)), color, -1)
        cv2.putText(out, label, (int(x1) + 3, int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), text_thick, cv2.LINE_AA)
    return out