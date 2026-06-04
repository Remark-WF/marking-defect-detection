# Конвертация весов PT -> ONNX -> RKNN

Для обучения используются веса `.pt`. Для Orange Pi 5 нужны веса `.rknn`, потому что инференс выполняется через NPU RK3588.

Общий путь:

```text
.pt -> .onnx -> .rknn
```

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
