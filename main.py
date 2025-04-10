import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from coco_parser import COCOParser
import os

coco_annotations_file="Solar Panel Fault Dataset.v1i.coco/train/_annotations.coco.json"
coco_images_dir="Solar Panel Fault Dataset.v1i.coco/train"
coco= COCOParser(coco_annotations_file, coco_images_dir)

print(coco.cat_dict.values())#all of classes contained in the dataset
# define a list of colors for drawing bounding boxes
def assign_colour(class_name):
    if class_name == "Physical Damage":
        return "purple"
    elif class_name =="Bird Drop":
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
color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "magenta","blue"]
num_imgs_to_disp = 4
total_images = len(coco.get_imgIds())  # total number of images
sel_im_idxs = np.random.permutation(total_images)[:num_imgs_to_disp]#randomly select the indexes of images to display
img_ids = coco.get_imgIds()
selected_img_ids = [img_ids[i] for i in sel_im_idxs]#select images with chosen early random indexes
ann_ids = coco.get_annIds(selected_img_ids)#select annotations
im_licenses = coco.get_imgLicenses(selected_img_ids)#select licenses
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))#creates a 2×2 grid of subplots to plot four images in a single figure.
ax = ax.ravel()
for i, im in enumerate(selected_img_ids):
    im_info = coco.im_dict[im]
    file_name = im_info['file_name']
    image_path = os.path.join(coco_images_dir, file_name)
    image = Image.open(image_path)
    ann_ids = coco.get_annIds(im)
    annotations = coco.load_anns(ann_ids)
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(b) for b in bbox]
        class_id = ann["category_id"]
        class_name = coco.load_cats(class_id)[0]["name"]
        license = coco.get_imgLicenses(im)[0]["name"]
        color_ = assign_colour(class_name)
        print(class_name)
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color_, facecolor='none')
        t_box = ax[i].text(x, y, class_name, color='red', fontsize=10)
        t_box.set_bbox(dict(boxstyle='square, pad=0', facecolor='white', alpha=0.6, edgecolor='blue'))
        ax[i].add_patch(rect)

    ax[i].axis('off')
    ax[i].imshow(image)
    ax[i].set_xlabel('Longitude')
    ax[i].set_title(f"License: {license}")
plt.tight_layout()
plt.show()