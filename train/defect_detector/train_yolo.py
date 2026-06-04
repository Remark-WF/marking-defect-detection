import os
import argparse
from ultralytics import YOLO
import torch
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение второй модели YOLOv8 (детекция 4 классов)"
    )
    parser.add_argument("--data", type=str,
                        default=r"C:\Users\Mikhail\Diplom\train_defects\dataset\data.yaml",
                        help="Путь к data.yaml")
    parser.add_argument("--model", type=str,
                        default="yolov8n.pt",
                        help="Предобученная модель")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Количество эпох")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Размер изображения")
    parser.add_argument("--batch", type=int, default=8,
                        help="Размер мини-пакета")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cuda:0 или cpu")
    parser.add_argument("--name", type=str,
                        default="train_defects",
                        help="Имя запуска")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("🚀 ЗАПУСК ОБУЧЕНИЯ ВТОРОЙ МОДЕЛИ (ДЕТЕКЦИЯ 4 КЛАССОВ)")
    print("=" * 60)
    print(f"📂 Датасет: {args.data}")
    print(f"🧠 Модель: {args.model}")
    print(f"📏 Размер изображения: {args.img_size}")
    print(f"📦 Batch size: {args.batch}")
    print(f"🔁 Эпохи: {args.epochs}")
    print(f"💻 Устройство: {args.device}")
    print("=" * 60)

    if not os.path.exists(args.data):
        print(f"❌ Файл {args.data} не найден!")
        return

    torch.manual_seed(42)

    model = YOLO(args.model)

    start_time = time.time()

    # Обучение
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        device=args.device,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        patience=10,
        project="runs/detect",
        name=args.name,
        exist_ok=True,
        seed=42
    )

    train_time = time.time() - start_time

    print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"⏱ Время обучения: {train_time/60:.2f} минут")
    print(f"📁 Результаты сохранены в: runs/detect/{args.name}/")

    # Финальная валидация
    print("\n📊 Запуск финальной валидации best.pt...")

    best_model_path = f"runs/detect/{args.name}/weights/best.pt"
    model = YOLO(best_model_path)

    metrics = model.val()

    print("\n" + "=" * 60)
    print("📈 ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 60)

    print(f"Precision:       {metrics.box.mp:.4f}")
    print(f"Recall:          {metrics.box.mr:.4f}")
    print(f"mAP@0.5:         {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:    {metrics.box.map:.4f}")

    print("=" * 60)
    print("🎯 Графики находятся в папке запуска.")
    print("=" * 60)


if __name__ == "__main__":
    main()