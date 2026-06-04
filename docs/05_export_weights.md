# Конвертация весов PT -> ONNX -> RKNN

Для обучения используются веса `.pt`. Для Orange Pi 5 нужны веса `.rknn`, потому что инференс выполняется через NPU RK3588.

Общий путь:

```text
.pt -> .onnx -> .rknn
```

## Внешняя документация

Перед настройкой окружения полезно открыть первоисточники:

- Ultralytics export modes: https://docs.ultralytics.com/modes/export/
- Ultralytics ONNX integration: https://docs.ultralytics.com/integrations/onnx/
- Rockchip RKNN Toolkit2: https://github.com/airockchip/rknn-toolkit2
- Rockchip RKNN Model Zoo: https://github.com/airockchip/rknn_model_zoo
- ONNX Python API: https://onnx.ai/onnx/api/

## Где выполнять конвертацию

Разделите две среды:

```text
Компьютер / Linux x86_64:
  .pt -> .onnx -> .rknn
  нужен rknn-toolkit2

Orange Pi 5:
  запуск .rknn
  нужен rknn-lite2
```

`rknn-toolkit2` нужен для сборки RKNN-моделей. `rknn-lite2` нужен для запуска готовых RKNN-моделей на плате. Это разные пакеты.

## Рекомендуемая среда для rknn-toolkit2

Самый предсказуемый вариант - Linux x86_64 с Python 3.10, потому что wheel-файлы RKNN Toolkit2 обычно привязаны к конкретной версии Python и архитектуре.

Пример:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip git
python3.10 -m venv .venv-rknn
source .venv-rknn/bin/activate
pip install --upgrade pip
```

Дальше установите wheel `rknn_toolkit2-*.whl`, который подходит под вашу версию Python:

```bash
pip install -r requirements-rknn-convert.txt
pip install rknn_toolkit2-*.whl
```

Проверка:

```bash
python -c "from rknn.api import RKNN; print('RKNN Toolkit2 OK')"
```

Если wheel лежит во внешнем хранилище проекта, скачайте его в папку конвертации. Если wheel берется из официального репозитория Rockchip, сверяйте версию Python в имени файла, например `cp310` означает Python 3.10.

## 1. Экспорт PT в ONNX

Скрипт:

```text
export/export_onnx.py
```

Перед запуском положите `.pt` веса в рабочую папку, например:

```text
pt/
  label_detector.pt
  label_defects_detector.pt
```

Запуск:

```bash
python export/export_onnx.py
```

На выходе должны появиться ONNX-файлы:

```text
onnx/
  label_detector.onnx
  label_defects_detector.onnx
```

Если скрипт ожидает другие имена, измените список моделей внутри `export/export_onnx.py`.

Важные параметры:

```python
EXPORT_CONFIG = {
    "imgsz": 640,
    "opset": 12,
    "dynamic": False,
    "simplify": True,
    "batch": 1
}
```

Что они означают:

- `imgsz` должен соответствовать размеру, с которым вы планируете запускать модель.
- `opset` лучше менять только при необходимости.
- `dynamic=False` делает статический вход, это проще для RKNN.
- `simplify=True` упрощает ONNX-граф.
- `batch=1` подходит для потокового инференса на Orange Pi.

## 2. Проверка ONNX

Перед конвертацией в RKNN полезно проверить, что ONNX-файл существует и открывается:

```bash
python -c "import onnx; onnx.checker.check_model('onnx/label_detector.onnx'); print('OK')"
```

Если пакета `onnx` нет:

```bash
pip install onnx
```

## 3. Конвертация ONNX в RKNN

Конвертация выполняется на Linux x86_64 с установленным `rknn-toolkit2`. Это не то же самое, что `rknn-lite2` на Orange Pi.

Скрипт:

```text
export/export_rknn.py
```

Запуск:

```bash
python export/export_rknn.py
```

Внутри скрипта должен быть указан target:

```python
PLATFORM = "rk3588"
```

Для FP16 используется:

```python
rknn.build(do_quantization=False)
```

В конфигурации важно совпадение preprocessing:

```python
rknn.config(
    target_platform="rk3588",
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
)
```

В текущем пайплайне изображение подготавливается вручную: letterbox, BGR -> RGB, затем нормализация через настройки RKNN. Если изменить preprocessing в Python-коде, нужно проверить и настройки `rknn.config`.

Ожидаемый результат:

```text
rknn/
  label_detector_fp16.rknn
  label_defects_detector_fp16.rknn
```

## 4. Куда положить RKNN на Orange Pi

Скопируйте файлы в папку рядом с пайплайном или в директорию моделей:

```text
label_detector_fp16.rknn
label_defects_detector_fp16.rknn
```

В `src/pipeline/run_pipeline_orangepi.py` проверьте имена:

```python
LABEL_MODEL_PATH = "label_detector_fp16.rknn"
DEFECT_MODEL_PATH = "label_defects_detector_fp16.rknn"
```

## Частые проблемы

### RKNN не загружается на Orange Pi

Проверьте:

- модель сконвертирована именно под `rk3588`;
- используется `rknn-lite2`, а не `rknn-toolkit2`;
- имена файлов совпадают с константами в пайплайне;
- файлы действительно находятся в текущей рабочей папке.

### Результаты RKNN отличаются от PT

Проверьте preprocessing:

- размер входа модели;
- RGB/BGR;
- нормализацию;
- letterbox;
- порядок классов;
- thresholds и NMS.

### Нужна INT8-квантизация

INT8 может ускорить инференс, но требует calibration dataset. Для дипломной версии использовался более простой и стабильный FP16-вариант.

## Почему post-processing не экспортируется внутрь модели

В этом проекте ONNX/RKNN содержит только нейросетевую часть. Декодирование выходов YOLO, фильтрация по confidence, NMS и перевод координат обратно в исходный кадр выполняются вручную в `src/pipeline/rknn_utils.py`.

Причины:

- RKNN не всегда корректно переносит сложный post-processing из экспортированной YOLO-модели.
- У разных YOLO-версий отличается форма выхода: например `C=5`, `C=4+nc` или `C=5+nc`.
- Нужен контроль над threshold, NMS и логикой классов прямо на Orange Pi.
- Проще отлаживать расхождения между PyTorch/ONNX/RKNN.
- Для промышленного пайплайна удобнее отделить модель от бизнес-логики решения о браке.

Подробнее: [Почему post-processing написан вручную](11_manual_postprocessing.md).
