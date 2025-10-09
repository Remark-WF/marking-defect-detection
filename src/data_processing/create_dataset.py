import os
import shutil
import random
import yaml

# Исходные папки после экспорта из Label Studio
src_images = "export/images"  # Папка с изображениями
src_labels = "export/labels"  # Папка с метками

# Итоговая структура
out_dir = "dataset"
# Создаем необходимые директории для train и val
for split in ["train", "val"]:
    os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

# Список всех изображений в папке с изображениями
all_images = [f for f in os.listdir(src_images) if f.lower().endswith((".jpg", ".png"))]
random.shuffle(all_images)  # Перемешиваем изображения

# Делим данные на 80/20 для train и val
split_idx = int(0.8 * len(all_images))
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

def move_files(img_list, split):
    """
    Функция для копирования изображений и их соответствующих меток в нужные директории.
    """
    for f in img_list:
        name, _ = os.path.splitext(f)  # Получаем имя без расширения
        img_src = os.path.join(src_images, f)  # Путь к изображению
        lbl_src = os.path.join(src_labels, name + ".txt")  # Путь к метке

        img_dst = os.path.join(out_dir, "images", split, f)  # Путь для сохранения изображения
        lbl_dst = os.path.join(out_dir, "labels", split, name + ".txt")  # Путь для сохранения метки

        shutil.copy(img_src, img_dst)  # Копируем изображение
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)  # Копируем метку, если она существует

# Перемещаем изображения и метки в train и val
move_files(train_imgs, "train")
move_files(val_imgs, "val")

print(f"Готово! Train: {len(train_imgs)}, Val: {len(val_imgs)}")

# Создаем файл data.yaml для использования в моделях (например, YOLO)
data_yaml = {
    "path": out_dir,
    "train": "images/train",
    "val": "images/val",
    "names": {0: "Этикетка"}  # Здесь указываем название класса для меток
}

# Сохраняем файл data.yaml
yaml_path = os.path.join(out_dir, "data.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, allow_unicode=True)

print(f"Файл data.yaml создан: {yaml_path}")
