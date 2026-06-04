import os
import subprocess
from ultralytics import YOLO

EXPORT_CONFIG = {
    "imgsz": 640, #Важно не забывать менять значение исходя из параметров обучения
    "opset": 12,
    "dynamic": False,
    "simplify": True,
    "batch": 1
}


def export_yolov5(model_path):
    print(f"[YOLOv5] Exporting {model_path}")

    cmd = [
        "python", "yolov5/export.py",
        "--weights", model_path,
        "--img", str(EXPORT_CONFIG["imgsz"]),
        "--include", "onnx",
        "--opset", str(EXPORT_CONFIG["opset"]),
    ]

    if EXPORT_CONFIG["simplify"]:
        cmd.append("--simplify")

    subprocess.run(cmd, check=True)


def export_ultralytics(model_path):
    print(f"[YOLOv8/11] Exporting {model_path}")

    model = YOLO(model_path)

    model.export(
        format="onnx",
        imgsz=EXPORT_CONFIG["imgsz"],
        opset=EXPORT_CONFIG["opset"],
        dynamic=EXPORT_CONFIG["dynamic"],
        simplify=EXPORT_CONFIG["simplify"],
        batch=EXPORT_CONFIG["batch"]
    )


def export_model(model_path):
    if "yolov5" in model_path:
        export_yolov5(model_path)
    else:
        export_ultralytics(model_path)


if __name__ == "__main__":
    models = [
        "pt/label_defects_detector_yolov5n.pt",
        "pt/label_defects_detector_yolov8n.pt",
        "pt/label_defects_detector_yolov11n.pt"
    ]

    for m in models:
        if os.path.exists(m):
            export_model(m)
        else:
            print(f"❌ Not found: {m}")
