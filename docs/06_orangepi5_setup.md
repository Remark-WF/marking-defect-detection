# Настройка Orange Pi 5

Эта инструкция описывает базовую подготовку Orange Pi 5 для запуска RKNN-пайплайна.

## Железо

Рекомендуемый набор:

- Orange Pi 5 или совместимая плата на RK3588;
- блок питания с достаточным током;
- microSD/eMMC с Linux;
- USB-камера или камера, доступная как `/dev/video*`;
- USB-накопитель для записи видео, если используется сборщик.

## Базовая проверка системы

```bash
uname -a
lsb_release -a
ls /dev/video*
```

Обновление пакетов:

```bash
sudo apt update
sudo apt upgrade -y
```

Полезные пакеты:

```bash
sudo apt install -y python3 python3-pip python3-venv git ffmpeg v4l-utils
```

## Python-окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-orangepi.txt
```

Для RKNN-инференса нужен `rknn-lite2`, подходящий под вашу версию Python, Linux и RK3588. Обычно он устанавливается из wheel-файла, который нужно положить во внешнее хранилище проекта.

Пример:

```bash
pip install rknn_toolkit_lite2-*.whl
```

Проверка:

```bash
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')"
```

## Куда положить модели

Положите рядом с `run_pipeline_orangepi.py` или в рабочую папку:

```text
label_detector_fp16.rknn
label_defects_detector_fp16.rknn
```

Если модели лежат в другой папке, измените пути в `src/pipeline/run_pipeline_orangepi.py`.

## Проверка камеры

Список камер:

```bash
v4l2-ctl --list-devices
ls /dev/video*
```

Форматы камеры:

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

Быстрый снимок:

```bash
ffmpeg -f video4linux2 -i /dev/video0 -frames:v 1 test.jpg
```

## Проверка NPU

Если RKNNLite установлен правильно, пайплайн при запуске должен загрузить `.rknn` модели без ошибки `load_rknn failed`.

Типичные причины ошибки:

- модель сконвертирована не под `rk3588`;
- файл поврежден или не скопирован;
- установлен неподходящий wheel `rknn-lite2`;
- пути к моделям неверные.

## Рекомендации по производительности

- используйте FP16 RKNN как базовый стабильный вариант;
- обрабатывайте не каждый кадр, если поток слишком тяжелый;
- запускайте вторую модель только после найденной этикетки;
- сохраняйте не все crop, а только нужные для отладки;
- проверяйте FPS на реальном видео, а не только на одиночных картинках.
