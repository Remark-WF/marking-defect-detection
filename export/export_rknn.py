import os
from rknn.api import RKNN

PLATFORM = "rk3588"

# 👉 ты сам задаёшь, что конвертировать
MODELS = [
    "onnx/label_detector_yolov5n.onnx",
    "onnx/label_detector_yolov8n.onnx",
    "onnx/label_detector_yolov11n.onnx",
]

RKNN_DIR = "rknn"


def convert_model(onnx_path):
    name = os.path.splitext(os.path.basename(onnx_path))[0]
    rknn_path = os.path.join(RKNN_DIR, name + "_fp16.rknn")

    print(f"\n=== Converting: {onnx_path} ===")

    if not os.path.exists(onnx_path):
        print(f"❌ File not found: {onnx_path}")
        return

    rknn = RKNN(verbose=True)

    print("--> Config")
    rknn.config(
        target_platform=PLATFORM,
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
    )

    print("--> Load ONNX")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"❌ Load failed: {onnx_path}")
        return

    print("--> Build")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"❌ Build failed: {onnx_path}")
        return

    print("--> Export RKNN")
    os.makedirs(RKNN_DIR, exist_ok=True)
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"❌ Export failed: {onnx_path}")
        return

    rknn.release()
    print(f"✅ Saved: {rknn_path}")


def main():
    for model in MODELS:
        convert_model(model)


if __name__ == "__main__":
    main()
