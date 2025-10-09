import os
from ultralytics import YOLO
import argparse

# Функция для парсинга аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description="Мини-скрипт для запуска обучения YOLOv8 на своём датасете")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Путь к файлу data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Путь к предобученной модели YOLOv8")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох для обучения")
    parser.add_argument("--img_size", type=int, default=960, help="Размер изображений для обучения")
    parser.add_argument("--batch", type=int, default=8, help="Размер пакета (batch size)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Устройство для обучения (cuda или cpu)")

    return parser.parse_args()

def main():
    # Получаем аргументы командной строки
    args = parse_args()

    # Проверка наличия файла data.yaml
    if not os.path.exists(args.data):
        print(f"❌ Ошибка: Файл {args.data} не найден!")
        return

    print(f"✅ Используется датасет: {args.data}")
    
    # Загружаем модель
    model = YOLO(args.model)

    # Тренируем модель
    print("🚀 Запуск обучения...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        project="runs/detect",
        name="train_labels",
        exist_ok=True,
        device=args.device
    )

    print("\n✅ Обучение завершено!")
    print("Модель сохранена здесь: runs/detect/train_labels/weights/best.pt")

if __name__ == "__main__":
    main()
