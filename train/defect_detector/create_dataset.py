import os
import shutil
import random
import yaml

# === Настройки ===
src_images = "images"   # Папка с изображениями из Label Studio
src_labels = "labels"   # Папка с YOLO-метками (.txt)
out_dir = "dataset"     # Куда собирать итоговую структуру

# Классы под твою задачу
classes = [
    "anchor_lot",
    "anchor_icon",
    "anchor_volume",
    "target_field"
]

# Создаем директории
for split in ["train", "val"]:
    os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

# === Собираем список всех изображений ===
all_images = [f for f in os.listdir(src_images) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
all_images = [f for f in all_images if os.path.exists(os.path.join(src_labels, os.path.splitext(f)[0] + ".txt"))]

random.shuffle(all_images)

# === Делим на train / val (80 / 20) ===
split_idx = int(0.8 * len(all_images))
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

def move_files(img_list, split):
    """
    Копирует изображения и метки в соответствующие подпапки train/val.
    """
    for f in img_list:
        name, _ = os.path.splitext(f)
        img_src = os.path.join(src_images, f)
        lbl_src = os.path.join(src_labels, name + ".txt")

        img_dst = os.path.join(out_dir, "images", split, f)
        lbl_dst = os.path.join(out_dir, "labels", split, name + ".txt")

        # Копируем только если и изображение, и метка существуют
        if os.path.exists(img_src) and os.path.exists(lbl_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)

move_files(train_imgs, "train")
move_files(val_imgs, "val")

print(f"✅ Датасет собран!")
print(f"Train: {len(train_imgs)} изображений")
print(f"Val: {len(val_imgs)} изображений")

# === Создаем data.yaml для YOLO ===
data_yaml = {
    "path": out_dir,  # корень датасета
    "train": "images/train",
    "val": "images/val",
    "nc": len(classes),
    "names": {i: name for i, name in enumerate(classes)}
}

yaml_path = os.path.join(out_dir, "data.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, allow_unicode=True)

print(f"📄 Файл data.yaml создан: {yaml_path}")
