import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from coco_parser import COCOParser
import matplotlib.patches as patches


# SolarPanelDataset class to load and preprocess images and annotations
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

        # generator function to process images and their annotations
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


def feature_extractor():
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(180, 180, 3)))
    return base_model


def extract_features(dataset):
    base_model = feature_extractor()
    feature_list = []
    for img, _ in dataset:
        img = tf.expand_dims(img, axis=0)
        features = base_model(img)
        feature_list.append(features.numpy())

    return np.array(feature_list)


for idx, (img, ann) in enumerate(train_ds):
    if idx == 4:
        original_img = img.numpy()
        boxes = ann['boxes'].numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(original_img)

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        plt.title("5th Image with Fault Bounding Boxes")
        plt.show()

        features_single = feature_extractor()(img[tf.newaxis, ...])

        def visualize_with_original(original_img, features, num_channels=63):
            feature_map = features[0]
            num_channels = min(num_channels, feature_map.shape[-1])

            plt.figure(figsize=(12, 10))
            plt.subplot(9, 8, 1)
            plt.imshow(original_img)
            plt.title("Original Image")
            plt.axis('off')

            for i in range(num_channels):
                plt.subplot(9, 8, i + 2)
                plt.imshow(feature_map[:, :, i], cmap='viridis')
                plt.axis('off')

            plt.tight_layout()
            plt.show()

        visualize_with_original(original_img, features_single, num_channels=63)
        break