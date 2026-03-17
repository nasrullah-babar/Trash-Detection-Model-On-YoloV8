import os
import random
from shutil import copyfile

dataset_path = r'' # Path to the dataset created in conv_json.py in root folder
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)


batch_dirs = [os.path.join(dataset_path, f'batch_{i}') for i in range(1, 16)]

images = []
for batch_dir in batch_dirs:
    if os.path.exists(batch_dir):
        images.extend([os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.endswith('.jpg')])

random.shuffle(images)
split_ratio = 0.8
split_index = int(len(images) * split_ratio)

train_images = images[:split_index]
val_images = images[split_index:]

def find_annotation_file(image_path):
    annotation_path = image_path.replace('.jpg', '.txt')
    if os.path.exists(annotation_path):
        return annotation_path
    return None

def copy_files(image_list, target_path):
    for img_path in image_list:
        annotation_path = find_annotation_file(img_path)

        if annotation_path:
            target_img_path = os.path.join(target_path, os.path.basename(img_path))
            target_annotation_path = os.path.join(target_path, os.path.basename(annotation_path))

            copyfile(img_path, target_img_path)
            copyfile(annotation_path, target_annotation_path)
        else:
            print(f"Warning: Annotation file for {img_path} not found. Skipping.")

copy_files(train_images, train_path)

copy_files(val_images, val_path)
