# rknn_models.py
import time
import numpy as np
from rknnlite.api import RKNNLite

from rknn_utils import letterbox_rgb, decode_yolo_like_1xCxN


class RKNNDetector:
    """
    Универсальная обёртка над RKNNLite для YOLO-подобных моделей.

    Поддерживаемые варианты выхода:
      - (1,5,N):  [x,y,w,h,conf]               (1 класс)
      - (1,4+nc,N): [x,y,w,h, cls_scores...]   (без obj)  <-- твой defects: (1,8,8400) при nc=4
      - (1,5+nc,N): [x,y,w,h,obj, cls_scores...]
    """

    def __init__(
        self,
        model_path: str,
        input_size: int,
        conf_th: float,
        iou_th: float,
        *,
        num_classes: int | None = None,
        has_obj: bool | None = None,
        box_format: str = "xywh",
        core_mask=None,
        debug_first_infer: bool = False,
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_th = conf_th
        self.iou_th = iou_th

        # Декодер-специфичные параметры
        self.num_classes = num_classes
        self.has_obj = has_obj
        self.box_format = box_format

        # RKNN runtime
        self.core_mask = core_mask if core_mask is not None else RKNNLite.NPU_CORE_0_1_2
        self.rknn = None

        # Debug
        self._debug_first_infer = bool(debug_first_infer)
        self._debug_printed = False

    def open(self):
        """Загрузка модели и инициализация рантайма."""
        self.rknn = RKNNLite(verbose=False)

        ret = self.rknn.load_rknn(self.model_path)
        print(f"  Код возврата load_rknn: {ret}")
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {self.model_path}")

        ret = self.rknn.init_runtime(async_mode=True, core_mask=self.core_mask)
        if ret != 0:
            raise RuntimeError("init_runtime failed")

    def close(self):
        """Освобождение ресурсов."""
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def infer(self, frame_bgr):
        """
        Returns:
          dets: list (x1,y1,x2,y2, score, class_id) in ORIGINAL coords
          dt_ms: inference time (ms)
          out_shape: raw output shape
        """
        if self.rknn is None:
            raise RuntimeError("RKNNDetector is not opened. Call open() first.")

        h0, w0 = frame_bgr.shape[:2]

        # Preprocess: letterbox + BGR->RGB, uint8
        rgb, scale, pad_left, pad_top = letterbox_rgb(frame_bgr, self.input_size)
        inp = np.expand_dims(rgb, axis=0)  # NHWC uint8

        # Inference
        t0 = time.time()
        outs = self.rknn.inference(inputs=[inp])
        dt_ms = (time.time() - t0) * 1000.0

        if not isinstance(outs, (list, tuple)) or len(outs) == 0:
            raise RuntimeError("Empty outputs from RKNN")

        out0 = outs[0]
        out_shape = np.asarray(out0).shape

        # Debug only once if requested
        debug_once = False
        if self._debug_first_infer and not self._debug_printed:
            debug_once = True
            self._debug_printed = True

        # Postprocess: decode + NMS + unletterbox
        dets = decode_yolo_like_1xCxN(
            out0,
            input_size=self.input_size,
            conf_th=self.conf_th,
            iou_th=self.iou_th,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            orig_w=w0,
            orig_h=h0,
            num_classes=self.num_classes,
            has_obj=self.has_obj,
            box_format=self.box_format,
            debug_once=debug_once,
        )

        return dets, dt_ms, out_shape