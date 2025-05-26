import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
from coco_parser import COCOParser
import matplotlib.patches as patches
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import hog
from skimage import exposure

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

def assign_colour(class_name):
    """Function returns the colour based on the category."""
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


def edge_detection(img):
    """Applies Canny edge detection on the image."""
    img_uint8 = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    gray = cv2.cvtColor(img_uint8.numpy(), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_colored


def flatten_image(image):
    """Flatten the 2D image to a 1D vector."""
    return image.flatten()

def resize_to_match(image, target_shape):
    """Resize the image to match the target shape."""
    return cv2.resize(image, (target_shape[1], target_shape[0]))  # (height, width)

def extract_hog_image(img):
    """Extract and return the HOG visualization image from the RGB image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image

def extract_sift_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw the keypoints on the image (for visualization)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_with_keypoints

def extract_orb_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_with_keypoints

def extract_fast_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detect keypoints using FAST
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)

    # Compute descriptors using BRIEF
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints, descriptors = brief.compute(gray, keypoints)

    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_with_keypoints


def visualize_edge_features(dataset, num_images=4):
    data_list = list(dataset.as_numpy_iterator())
    idxs = np.random.choice(len(data_list), num_images, replace=False)

    fig, axs = plt.subplots(num_images, 2, figsize=(12, 4 * num_images))

    for i, idx in enumerate(idxs):
        img, targets = data_list[idx]
        boxes = targets['boxes']
        label_names = targets['label_names']

        img_np = (img * 255).astype(np.uint8)
        edge_img = edge_detection(img)

        for col, image in enumerate([img_np, edge_img]):
            ax = axs[i, col] if num_images > 1 else axs[col]
            ax.imshow(image)
            ax.axis('off')
            title = 'Original Image' if col == 0 else 'Edge Map'
            ax.set_title(title)

            for box, label in zip(boxes, label_names):
                x_min, y_min, x_max, y_max = box
                width, height = x_max - x_min, y_max - y_min
                color = assign_colour(label.decode('utf-8'))
                rect = patches.Rectangle((x_min, y_min), width, height,
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min - 5, label.decode('utf-8'), fontsize=8, color='black',
                        bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2'))

    plt.tight_layout()
    plt.show()

visualize_edge_features(train_ds, num_images=4)

def visualize_hog_features(dataset, num_images=4):
    data_list = list(dataset.as_numpy_iterator())
    idxs = np.random.choice(len(data_list), num_images, replace=False)

    fig, axs = plt.subplots(num_images, 2, figsize=(12, 4 * num_images))

    for i, idx in enumerate(idxs):
        img, targets = data_list[idx]
        boxes = targets['boxes']
        label_names = targets['label_names']

        img_np = (img * 255).astype(np.uint8)
        hog_img = extract_hog_image(img_np)

        for col, image in enumerate([img_np, hog_img]):
            ax = axs[i, col] if num_images > 1 else axs[col]
            cmap = None if col == 0 else 'gray'
            ax.imshow(image, cmap=cmap)
            ax.axis('off')
            ax.set_title('Original Image' if col == 0 else 'HOG Visualization')

            for box, label in zip(boxes, label_names):
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                color = assign_colour(label.decode('utf-8'))
                rect = patches.Rectangle((x_min, y_min), width, height,
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min - 5, label.decode('utf-8'), fontsize=8, color='black',
                        bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2'))

    plt.tight_layout()
    plt.show()

visualize_hog_features(train_ds, num_images=4)


def visualize_sift_features(dataset, num_images=4):
    data_list = list(dataset.as_numpy_iterator())
    idxs = np.random.choice(len(data_list), num_images, replace=False)

    fig, axs = plt.subplots(num_images, 2, figsize=(12, 4 * num_images))

    for i, idx in enumerate(idxs):
        img, targets = data_list[idx]
        boxes = targets['boxes']
        label_names = targets['label_names']

        img_np = (img * 255).astype(np.uint8)
        sift_img = extract_sift_features(img_np)

        for col, image in enumerate([img_np, sift_img]):
            ax = axs[i, col] if num_images > 1 else axs[col]
            ax.imshow(image)
            ax.axis('off')
            title = 'Original Image' if col == 0 else 'SIFT Keypoints'
            ax.set_title(title)

            for box, label in zip(boxes, label_names):
                x_min, y_min, x_max, y_max = box
                width, height = x_max - x_min, y_max - y_min
                color = assign_colour(label.decode('utf-8'))
                rect = patches.Rectangle((x_min, y_min), width, height,
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min - 5, label.decode('utf-8'), fontsize=8, color='black',
                        bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2'))

    plt.tight_layout()
    plt.show()

visualize_sift_features(train_ds, num_images=4)

def visualize_orb_features(dataset, num_images=4):
    data_list = list(dataset.as_numpy_iterator())
    idxs = np.random.choice(len(data_list), num_images, replace=False)

    fig, axs = plt.subplots(num_images, 2, figsize=(12, 4 * num_images))

    for i, idx in enumerate(idxs):
        img, targets = data_list[idx]
        boxes = targets['boxes']
        label_names = targets['label_names']

        img_np = (img * 255).astype(np.uint8)
        orb_img = extract_orb_features(img_np)

        for col, image in enumerate([img_np, orb_img]):
            ax = axs[i, col] if num_images > 1 else axs[col]
            ax.imshow(image)
            ax.axis('off')
            ax.set_title('Original Image' if col == 0 else 'ORB Features')

            for box, label in zip(boxes, label_names):
                x_min, y_min, x_max, y_max = box
                width, height = x_max - x_min, y_max - y_min
                color = assign_colour(label.decode('utf-8'))
                rect = patches.Rectangle((x_min, y_min), width, height,
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min - 5, label.decode('utf-8'), fontsize=8, color='black',
                        bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2'))

    plt.tight_layout()
    plt.show()

visualize_orb_features(train_ds, num_images=4)

def visualize_fast_features(dataset, num_images=4):
    data_list = list(dataset.as_numpy_iterator())
    idxs = np.random.choice(len(data_list), num_images, replace=False)

    fig, axs = plt.subplots(num_images, 2, figsize=(12, 4 * num_images))

    for i, idx in enumerate(idxs):
        img, targets = data_list[idx]
        boxes = targets['boxes']
        label_names = targets['label_names']

        img_np = (img * 255).astype(np.uint8)
        fast_img = extract_fast_features(img_np)

        for col, image in enumerate([img_np, fast_img]):
            ax = axs[i, col] if num_images > 1 else axs[col]
            ax.imshow(image)
            ax.axis('off')
            ax.set_title('Original Image' if col == 0 else 'FAST + BRIEF Features')

            for box, label in zip(boxes, label_names):
                x_min, y_min, x_max, y_max = box
                width, height = x_max - x_min, y_max - y_min
                color = assign_colour(label.decode('utf-8'))
                rect = patches.Rectangle((x_min, y_min), width, height,
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min - 5, label.decode('utf-8'), fontsize=8, color='black',
                        bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2'))

    plt.tight_layout()
    plt.show()

visualize_fast_features(train_ds, num_images=4)

