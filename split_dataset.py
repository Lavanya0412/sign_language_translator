import os
import shutil
import random

# ------------------- CONFIGURATION -------------------
# Your original collected data
original_dir = r"E:\test_data_2.0\sign2text_dataset_3.0\AtoZ_3.0"
# New directory where split folders will be created (train and val)
split_dir = r"E:\test_data_2.0\sign2text_dataset_3.0_split"
train_ratio = 0.85  # 85% train, 15% validation

# ------------------- CREATE FOLDER STRUCTURE -------------------
train_dir = os.path.join(split_dir, "train")
val_dir = os.path.join(split_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Create class folders (A-Z) inside train and val
for cls in sorted(os.listdir(original_dir)):
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

# ------------------- SPLIT AND COPY IMAGES -------------------
for cls in sorted(os.listdir(original_dir)):
    cls_path = os.path.join(original_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    n_train = int(len(images) * train_ratio)
    train_images = images[:n_train]
    val_images = images[n_train:]

    # Copy train images
    for img in train_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(train_dir, cls, img)
        shutil.copy(src, dst)

    # Copy validation images
    for img in val_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(val_dir, cls, img)
        shutil.copy(src, dst)

    print(f"{cls}: {len(train_images)} train, {len(val_images)} val")

print("\n✅ Dataset split completed!")
print(f"Train folder: {train_dir}")
print(f"Validation folder: {val_dir}")