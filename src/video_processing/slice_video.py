import cv2
import os

# Указываем путь к видеофайлу, который нужно нарезать
VIDEO_PATH = "path_to_your_video_file.mp4"  # Путь к вашему видеофайлу
OUT_DIR = "output_frames"  # Папка для сохранения кадров
os.makedirs(OUT_DIR, exist_ok=True)

# Частота выборки: каждые 3 кадра из 30 FPS, чтобы не создавать дубликаты
SAMPLE_EVERY = 3

def is_blurry(gray, thr=60.0):
    """
    Проверяет, является ли кадр размытым, используя Лапласов оператор.
    Чем меньше результат, тем сильнее смаз.
    """
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < thr

# Открытие видео
cap = cv2.VideoCapture(VIDEO_PATH)

i = 0
saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % SAMPLE_EVERY == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not is_blurry(gray):
            h, w = frame.shape[:2]
            # Применяем CLAHE для выравнивания контраста, что повышает читаемость кадра
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            frame_yuv[:, :, 0] = clahe.apply(frame_yuv[:, :, 0])
            frame_eq = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)

            # Сохраняем кадр с уникальным именем
            out_path = os.path.join(OUT_DIR, f"frame_{saved:06d}.jpg")
            cv2.imwrite(out_path, frame_eq, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
    i += 1

cap.release()
print(f"Saved frames: {saved}")
