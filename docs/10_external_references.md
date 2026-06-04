# Внешняя документация

Этот файл нужен, чтобы следующий разработчик быстро нашел первоисточники по плате, YOLO, ONNX и RKNN.

## Orange Pi 5

- Orange Pi 5, официальная страница: http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-5.html
- Orange Pi Wiki: http://www.orangepi.org/orangepiwiki/
- Раздел загрузок Orange Pi: http://www.orangepi.org/html/serviceAndSupport/index.html

Что искать:

- образ ОС для Orange Pi 5;
- `User Manual`;
- инструкции по записи образа;
- логин и пароль по умолчанию для выбранного образа;
- настройку SSH, Wi-Fi/Ethernet и камеры.

## Запись образа ОС

- balenaEtcher: https://etcher.balena.io/
- Raspberry Pi Imager: https://www.raspberrypi.com/software/

Эти программы удобны для записи `.img` образов на microSD/eMMC.

## YOLO / Ultralytics

- Ultralytics docs: https://docs.ultralytics.com/
- Train mode: https://docs.ultralytics.com/modes/train/
- Val mode: https://docs.ultralytics.com/modes/val/
- Predict mode: https://docs.ultralytics.com/modes/predict/
- Export mode: https://docs.ultralytics.com/modes/export/
- ONNX integration: https://docs.ultralytics.com/integrations/onnx/

## ONNX

- ONNX documentation: https://onnx.ai/
- ONNX Python API: https://onnx.ai/onnx/api/

## Rockchip RKNN

- RKNN Toolkit2: https://github.com/airockchip/rknn-toolkit2
- RKNN Model Zoo: https://github.com/airockchip/rknn_model_zoo

Важно различать:

- `rknn-toolkit2` - конвертация и сборка моделей на Linux x86_64;
- `rknn-lite2` - запуск готовых `.rknn` моделей на плате RK3588.

## Linux и камера

- v4l-utils: https://linuxtv.org/wiki/index.php/V4l-utils
- FFmpeg: https://ffmpeg.org/documentation.html
- OpenCV documentation: https://docs.opencv.org/

## Как использовать эти ссылки в дипломе

Для ВКР лучше ссылаться не только на блог-посты, но и на:

- официальную документацию Orange Pi;
- официальную документацию Ultralytics;
- репозитории Rockchip RKNN;
- документацию ONNX;
- документацию OpenCV/FFmpeg для технической реализации захвата и обработки видео.
