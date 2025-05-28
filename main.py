
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
from coco_parser import COCOParser
import matplotlib.patches as patches


#SolarPanelDataset class to load and preprocess images and annotations
class SolarPanelDataset(tf.data.Dataset):
    def __new__(cls, images_dir, annotation_file):
        coco = COCOParser(annotation_file, images_dir)
        img_ids = coco.get_imgIds()
        image_paths = []
        bboxes = []
        labels = []
        label_names = []
        for img_id in img_ids:
            img_info = coco.im_dict[img_id]  # image information
            filename = img_info['file_name']  # image file name
            image_path = str(pathlib.Path(images_dir) / filename)  # image path
            image_paths.append(image_path)
            anns = coco.annIm_dict[img_id]  # annotations for the image
            img_bboxes = []
            img_labels = []
            img_label_names = []
            for ann in anns:
                x, y, width, height = ann['bbox']  # bbox coordinates
                x_min = x
                y_min = y
                x_max = x + width
                y_max = y + height
                img_bboxes.append([x_min, y_min, x_max, y_max])
                # class id and name from annotation
                class_id = ann["category_id"]
                class_name = coco.load_cats(class_id)[0]["name"]
                img_labels.append(class_id)
                img_label_names.append(class_name)

            # bounding boxes, labels, and names appended to the corresponding lists
            bboxes.append(np.array(img_bboxes, dtype=np.float32))
            labels.append(np.array(img_labels, dtype=np.int32))
            label_names.append(np.array(img_label_names, dtype=object))

        def generator():
            for img_path, bbox, label, label_name in zip(image_paths, bboxes, labels, label_names):
                img = tf.io.read_file(img_path)  # read image file
                img = tf.image.decode_jpeg(img, channels=3)  # decode image as RGB
                original_shape = tf.shape(img)[:2]  # original height and width
                img = tf.image.resize(img, (180, 180))  # resize image to 180x180
                img = tf.cast(img, tf.float32) / 255.0  # normalize image to [0, 1]
                # scaling factors for bounding boxes
                scale_y = 180.0 / tf.cast(original_shape[0], tf.float32)
                scale_x = 180.0 / tf.cast(original_shape[1], tf.float32)
                # scale bounding boxes to match the resized image
                bbox = tf.convert_to_tensor(bbox)
                bbox_scaled = bbox * [scale_x, scale_y, scale_x, scale_y]
                yield img, {'boxes': bbox_scaled, 'labels': label, 'label_names': label_name}

        # output signature of the dataset that specifies the structure of the data that will be yielded by the generator
        output_signature = (
            tf.TensorSpec(shape=(180, 180, 3), dtype=tf.float32),
            {
                'boxes': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                'labels': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'label_names': tf.TensorSpec(shape=(None,), dtype=tf.string),
            }
        )
        return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

train_ds = SolarPanelDataset(
    images_dir="Solar Panel Fault Dataset.v1i.coco/train",
    annotation_file="Solar Panel Fault Dataset.v1i.coco/train/_annotations.coco.json"
)

def assign_colour(class_name):
    if class_name == "Physical Damage":
        return "purple"

    elif class_name == "Bird Drop":
        return "yellow"
    elif class_name == "Non Defective":
        return "green"
    elif class_name == "Defective":
            return "red"
    elif class_name == "Dust":
            return "grey"
    elif class_name == "Dusty":
            return "orange"
    elif class_name == "Snow":
            return "pink"
    else:
            return "blue"

# function to visualize a random selection of images and their bounding boxes
def visualize_random_samples(dataset, num_images=4):
    data_list = list(dataset.as_numpy_iterator())
    # randomly pick 4 samples
    idxs = np.random.choice(len(data_list), num_images, replace=False)
    # subplots for displaying images
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.ravel()
    for i, idx in enumerate(idxs):
        img, targets = data_list[idx]
        boxes = targets['boxes']
        labels = targets['labels']
        label_names = targets['label_names']
        axs[i].imshow(img)
        axs[i].axis('off')
        for box, label_name in zip(boxes, label_names):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            color = assign_colour(label_name.decode('utf-8'))
            # rectangle for the bounding box
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            axs[i].add_patch(rect)
            t_box= axs[i].text(
                x_min,
                y_min - 5,
                label_name.decode('utf-8'),
                color='red',
                fontsize=8,
                backgroundcolor='white'
            )
            t_box.set_bbox(dict(boxstyle='square, pad=0', facecolor='white', alpha=0.6,
                                edgecolor='blue'))
    plt.tight_layout()
    plt.show()

visualize_random_samples(train_ds, num_images=4)