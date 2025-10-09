import cv2
import argparse
from ultralytics import YOLO
import os

# Пороги и настройки
CONF = 0.4  # порог уверенности

def parse_args():
    """
    Функция для парсинга аргументов командной строки
    """
    parser = argparse.ArgumentParser(description="Тестирование обученной YOLO модели на видео")
    parser.add_argument('--model', type=str, required=True, help="Путь к обученной модели (например, 'runs/detect/train_labels/weights/best.pt')")
    parser.add_argument('--video_in', type=str, required=True, help="Путь к входному видео")
    parser.add_argument('--video_out', type=str, default='result.mp4', help="Путь к выходному видео")
    parser.add_argument('--conf', type=float, default=CONF, help="Порог уверенности для предсказаний (0-1)")

    return parser.parse_args()

def main():
    # Парсим аргументы командной строки
    args = parse_args()

    # Проверка существования модели
    if not os.path.exists(args.model):
        print(f"❌ Ошибка: Модель по пути '{args.model}' не найдена!")
        return

    # Проверка существования входного видео
    if not os.path.exists(args.video_in):
        print(f"❌ Ошибка: Видео по пути '{args.video_in}' не найдено!")
        return

    # Загрузка модели
    model = YOLO(args.model)

    # Открытие видео
    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        print("❌ Ошибка при открытии видео файла!")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создание объекта для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_out, fourcc, fps, (w, h))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Предсказание на кадре
        results = model.predict(frame, conf=args.conf, verbose=False)
        annotated = results[0].plot()  # Получаем картинку с нарисованными детекциями

        # Запись результата в файл
        out.write(annotated)

        # Отображаем результат
        cv2.imshow("YOLO detections", annotated)

        # Выход из программы по нажатию ESC
        if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
            break

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Готово! Результат сохранён в {args.video_out}")

if __name__ == "__main__":
    main()
