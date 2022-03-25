import os
import json
import random
import shutil

from tqdm import tqdm

IMG_DIR = "coco/images"
ANN_DIR = "coco/annotations"
TARGET_DIR = "datasets/clothing_detection"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

image_names = os.listdir(IMG_DIR)

random.shuffle(image_names)

train = image_names[:int(len(image_names) * TRAIN_SPLIT)]
val = image_names[int(len(image_names) * TRAIN_SPLIT):]

print(len(train), len(val))

# Loading annotations

print("loading annotations")
annotations = json.load(open(os.path.join(ANN_DIR, "instances_all.json"), "r"))
print("successfully loaded annotations")

train_json = {"images": [], "annotations": [], "categories": []}
val_json = {"images": [], "annotations": [], "categories": []}

def get_filename(file):
    return file["file_name"]

def get_imageid(file):
    return file["image_id"]

def get_id(file):
    return file["id"]

if os.path.exists(os.path.join(TARGET_DIR, "train")):
    shutil.rmtree(os.path.join(TARGET_DIR, "train"))
if os.path.exists(os.path.join(TARGET_DIR, "val")):
    shutil.rmtree(os.path.join(TARGET_DIR, "val"))

os.mkdir(os.path.join(TARGET_DIR, "train"))
os.mkdir(os.path.join(TARGET_DIR, "val"))

for _, ann in tqdm(enumerate(annotations["images"]), total=len(annotations["annotations"])):
    if ann["file_name"] in train:
        train_json["images"].append(ann)
        shutil.copy(os.path.join(IMG_DIR, ann["file_name"]), os.path.join(TARGET_DIR, "train", ann["file_name"]))
    elif ann["file_name"] in val:
        val_json["images"].append(ann)
        shutil.copy(os.path.join(IMG_DIR, ann["file_name"]), os.path.join(TARGET_DIR, "val", ann["file_name"]))


train_ids = [get_id(file) for file in train_json["images"]]
val_ids = [get_id(file) for file in val_json["images"]]

for _, ann in tqdm(enumerate(annotations["annotations"]), total=len(annotations["annotations"])):
    if ann["image_id"] in train_ids:
        train_json["annotations"].append(ann)
    elif ann["image_id"] in val_ids:
        val_json["annotations"].append(ann)

print(len(train_json["images"]), len(train_json["annotations"]))
print(len(val_json["images"]), len(val_json["annotations"]))

train_json["categories"] = annotations["categories"]
val_json["categories"] = annotations["categories"]

with open(TARGET_DIR + "/annotations/instances_train.json", "w") as f:
    json.dump(train_json, f)
with open(TARGET_DIR + "/annotations/instances_val.json", "w") as f:
    json.dump(val_json, f)  