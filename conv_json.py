import json
import os
import shutil


taco_path = r'' #taco dataset path
data_path = os.path.join(taco_path, 'data')
output_path = r'' #output dataset path
annotations_file = os.path.join(data_path, 'annotations.json')
os.makedirs(output_path, exist_ok=True)

with open(annotations_file) as f:
    data = json.load(f)


class_names = [category['name'] for category in data['categories']]
class_name_to_id = {name: i for i, name in enumerate(class_names)}


def convert_to_yolo_format(data):
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        image_info = next(item for item in data['images'] if item['id'] == image_id)
        file_name = image_info['file_name']
        category_id = class_name_to_id[data['categories'][annotation['category_id'] - 1]['name']]

        x, y, w, h = annotation['bbox']
        dw = 1.0 / image_info['width']
        dh = 1.0 / image_info['height']
        x_center = (x + w / 2.0) * dw
        y_center = (y + h / 2.0) * dh
        w *= dw
        h *= dh

        batch_dir = os.path.dirname(file_name)
        label_path = os.path.join(output_path, batch_dir, os.path.splitext(os.path.basename(file_name))[0] + '.txt')

        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        with open(label_path, 'a') as f:
            f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")

        shutil.copy(os.path.join(data_path, file_name), os.path.join(output_path, batch_dir))


convert_to_yolo_format(data)