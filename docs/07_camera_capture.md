# Камера и захват изображения

Этот файл содержит полезные команды для проверки камеры и записи видео.

## Список камер

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

## Поддерживаемые разрешения и FPS

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

## Снимок через ffmpeg

```bash
ffmpeg -f video4linux2 -i /dev/video0 -frames:v 1 frame.jpg
```

С указанием разрешения:

```bash
ffmpeg -f video4linux2 -video_size 1280x720 -i /dev/video0 -frames:v 1 frame_720p.jpg
```

## Запись короткого видео

```bash
ffmpeg -f video4linux2 -video_size 1280x720 -framerate 30 -i /dev/video0 -t 10 test_camera.mp4
```

## Просмотр потока

Если есть графическая среда:

```bash
ffplay /dev/video0
```

Или через OpenCV:

```bash
python3 -c "import cv2; cap=cv2.VideoCapture(0); ok,frame=cap.read(); print(ok, frame.shape if ok else None); cap.release()"
```

## Настройки камеры

Посмотреть доступные параметры:

```bash
v4l2-ctl --device=/dev/video0 --list-ctrls
```

Пример изменения экспозиции:

```bash
v4l2-ctl --device=/dev/video0 --set-ctrl=exposure_auto=1
v4l2-ctl --device=/dev/video0 --set-ctrl=exposure_absolute=120
```

Пример изменения фокуса, если камера поддерживает:

```bash
v4l2-ctl --device=/dev/video0 --set-ctrl=focus_auto=0
v4l2-ctl --device=/dev/video0 --set-ctrl=focus_absolute=30
```

## Сбор видео штатным скриптом

Файл:

```text
src/collector/collector.py
```

Назначение:

- найти USB-накопитель;
- найти камеру;
- записывать видео сегментами;
- сохранять материал для датасета.

Запуск:

```bash
python3 src/collector/collector.py
```

Перед использованием проверьте пути и настройки внутри файла под вашу плату и камеру.
