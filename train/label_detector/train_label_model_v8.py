import os
import argparse
from ultralytics import YOLO
import torch
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение YOLOv8 с подробным выводом метрик"
    )
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                        help="Путь к data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Предобученная модель")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Количество эпох")
    parser.add_argument("--img_size", type=int, default=320,
                        help="Размер изображения")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="cuda:0 или cpu")
    parser.add_argument("--name", type=str, default="yolov8n_run",
                        help="Имя запуска")

    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = os.path.abspath("runs/detect_yolov8n")

    print("=" * 60)
    print("🚀 ЗАПУСК ОБУЧЕНИЯ YOLOv8n")
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

    # Фиксируем seed для воспроизводимости
    torch.manual_seed(42)

    # Загружаем модель
    model = YOLO(args.model)

    start_time = time.time()

    # Обучение
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        device=args.device,
        project=project_dir,
        name=args.name,
        exist_ok=True,
        seed=42
    )

    train_time = time.time() - start_time

    print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"⏱ Время обучения: {train_time/60:.2f} минут")
    print(f"📁 Результаты сохранены в: {project_dir}\\{args.name}\\")

    # Валидация лучшей модели
    print("\n📊 Запуск финальной валидации на best.pt...")

    best_model_path = os.path.join(project_dir, args.name, "weights", "best.pt")
    model = YOLO(best_model_path)

    metrics = model.val(data=args.data, device=args.device)

    print("\n" + "=" * 60)
    print("📈 ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 60)

    print(f"Precision:       {metrics.box.mp:.4f}")
    print(f"Recall:          {metrics.box.mr:.4f}")
    print(f"mAP@0.5:         {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:    {metrics.box.map:.4f}")

    print("=" * 60)
    print("🎯 Графики и PR-кривые находятся в папке запуска.")
    print("=" * 60)


if __name__ == "__main__":
    main()
