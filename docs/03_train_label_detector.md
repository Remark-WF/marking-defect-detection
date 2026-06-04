# Обучение модели поиска этикетки

Эта модель ищет этикетку на полном кадре.

## Подготовка окружения

На Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements-dev.txt
```

На Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Подготовка данных

Датасет должен быть в YOLO-формате:

```text
label_dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```

Пример `data.yaml`:

```yaml
train: images/train
val: images/val
names:
  0: label
```

## Запуск обучения

Пример для YOLOv8n:

```bash
python train/label_detector/train_label_model_v8.py --data path/to/label_dataset/data.yaml --model yolov8n.pt --epochs 25 --img_size 960 --batch 8 --device 0
```

Пример для YOLOv5n:

```bash
python train/label_detector/train_label_model_v5.py --data path/to/label_dataset/data.yaml --model yolov5nu.pt --epochs 25 --img_size 960 --batch 8 --device 0
```

Пример для YOLO11n:

```bash
python train/label_detector/train_label_model_v11.py --data path/to/label_dataset/data.yaml --model yolo11n.pt --epochs 25 --img_size 960 --batch 8 --device 0
```

Если GPU нет, используйте:

```bash
--device cpu
```

## Где искать результат

После обучения Ultralytics сохраняет результаты в папке `runs/`.

Важные файлы:

- `weights/best.pt` - лучший вес;
- `weights/last.pt` - последний вес;
- `results.csv` - метрики по эпохам;
- `results.png` - график обучения;
- `confusion_matrix.png` - матрица ошибок;
- `val_batch*_pred.jpg` - примеры предсказаний.

Финальный вес переименуйте:

```text
label_detector.pt
```

И положите во внешнее хранилище в `weights/pt/`.

## Проверенные метрики текущей версии

Лучший сохраненный результат для модели поиска этикетки:

```text
precision: 0.9989
recall:    0.96226
mAP50:     0.9651
mAP50-95:  0.95437
```

После изменения датасета эти значения нужно пересчитать.
