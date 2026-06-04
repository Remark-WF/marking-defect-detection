# Датасеты и внешнее хранилище

В репозиторий не нужно загружать датасеты, веса и видео. Они быстро делают Git тяжелым и неудобным. Все большие файлы храните во внешнем облаке, а в репозитории оставляйте ссылку и описание структуры.

## Рекомендуемая структура внешнего хранилища

```text
marking-defect-assets/
  datasets/
    label_dataset.zip
    defect_dataset.zip
    test_videos.zip
  weights/
    pt/
      label_detector.pt
      label_defects_detector.pt
    onnx/
      label_detector.onnx
      label_defects_detector.onnx
    rknn/
      label_detector_fp16.rknn
      label_defects_detector_fp16.rknn
  results/
    label_training_results.zip
    defect_training_results.zip
    inference_examples.zip
  docs/
    diploma.pdf
    presentation.pdf
```

## Датасет поиска этикетки

Формат YOLO:

```text
label_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Пример `data.yaml`:

```yaml
train: images/train
val: images/val
names:
  0: label
```

Текущий сохраненный вариант датасета:

- `200` изображений train;
- `51` изображение val;
- один класс: `label`.

## Датасет элементов маркировки

Классы:

```text
anchor_icon
anchor_lot
anchor_volume
target_field
```

Желательная структура:

```text
defect_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Если датасет лежит одной папкой `images/` и `labels/`, его лучше заранее разделить на train/val. В проекте есть пример скрипта `train/defect_detector/create_dataset.py`.

## Что обязательно положить в облако

- исходные или тестовые видео;
- датасет поиска этикетки;
- датасет элементов маркировки;
- финальные `.pt` веса;
- финальные `.onnx` веса;
- финальные `.rknn` веса для Orange Pi 5;
- результаты обучения: `results.csv`, confusion matrix, PR/F1/P/R curves;
- несколько примеров инференса на видео или изображениях.

## Как назвать финальные веса

Рекомендуемые имена:

```text
label_detector.pt
label_defects_detector.pt
label_detector.onnx
label_defects_detector.onnx
label_detector_fp16.rknn
label_defects_detector_fp16.rknn
```

Такие имена уже ожидаются в пайплайне Orange Pi. Если используете другие имена, поменяйте константы в `src/pipeline/run_pipeline_orangepi.py`.
