# Обучение модели элементов маркировки

Эта модель анализирует crop этикетки и ищет элементы маркировки.

## Классы

```text
anchor_icon
anchor_lot
anchor_volume
target_field
```

Список классов лежит в `configs/defect_classes.txt`.

## Подготовка датасета

Рекомендуемый YOLO-формат:

```text
defect_dataset/
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
  0: anchor_icon
  1: anchor_lot
  2: anchor_volume
  3: target_field
```

Если изображения и labels лежат одной общей папкой, используйте `train/defect_detector/create_dataset.py` как основу для разделения на train/val.

## Запуск обучения

YOLOv8n:

```bash
python train/defect_detector/train_model_v8.py --data path/to/defect_dataset/data.yaml --model yolov8n.pt --epochs 50 --img_size 640 --batch 8 --device 0
```

YOLOv5n:

```bash
python train/defect_detector/train_model_v5.py --data path/to/defect_dataset/data.yaml --model yolov5nu.pt --epochs 50 --img_size 640 --batch 8 --device 0
```

YOLO11n:

```bash
python train/defect_detector/train_model_v11.py --data path/to/defect_dataset/data.yaml --model yolo11n.pt --epochs 50 --img_size 640 --batch 8 --device 0
```

## Результаты

После обучения сохраните:

- `weights/best.pt`;
- `results.csv`;
- `results.png`;
- `confusion_matrix.png`;
- `PR_curve.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`;
- изображения `val_batch*_pred.jpg`.

Финальный вес переименуйте:

```text
label_defects_detector.pt
```

И положите во внешнее хранилище в `weights/pt/`.

## Проверенные метрики текущей версии

Лучший сохраненный результат для модели элементов маркировки:

```text
precision: 0.99577
recall:    1.0
mAP50:     0.995
mAP50-95:  0.86786
```

Важно: эти метрики получены на текущем датасете. Если условия съемки, камера, освещение или тип этикетки меняются, модель нужно перепроверять на новых данных.

## Где менять логику брака

Логика решения находится в `src/pipeline/run_pipeline_orangepi.py`, функция `decide_defective`.

Если меняются классы или ожидаемое количество элементов, нужно обновить:

- `configs/defect_classes.txt`;
- `data.yaml` датасета;
- thresholds в пайплайне;
- функцию `decide_defective`.
