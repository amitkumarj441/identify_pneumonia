import os 
import sys
import random
import math
import numpy as np
import pandas as pd 
import glob 
import json
from matplotlib import pyplot as plt

import cv2
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
from pneumonia import PneumoniaDataset, PneumoniaConfig
from functions import parse_dataset

DATA_DIR = '../input/'
TRAIN_DIR = os.path.join(DATA_DIR, 'stage_1_train_images')
TEST_DIR = os.path.join(DATA_DIR, 'stage_1_test_images')

MODEL_DIR = '../model/Mask_RCNN'
COCO_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_coco.h5')
ORIG_SIZE = 1024

# Import Mask_RCNN
sys.path.append(os.path.join(MODEL_DIR))  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

config = PneumoniaConfig()
config.display()

# Parse dataset
annotations = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_train_labels.csv'))
image_fps, image_annotations = parse_dataset(TRAIN_DIR, annotations)

# Split dataset into training vs. validation dataset 
image_fps_list = list(image_fps)
random.seed(42)
random.shuffle(image_fps_list)

val_size = 1500
image_fps_train = image_fps_list[val_size:]
image_fps_val = image_fps_list[:val_size]

print('train_size=', len(image_fps_train))
print('validation_size=', (len(image_fps_val)))

df = pd.DataFrame(image_fps_train, columns=['image_fps_train'])
df.to_csv('fps_train.csv', index=False)
df = pd.DataFrame(image_fps_val, columns=['image_fps_val'])
df.to_csv('fps_val.csv', index=False)

# training dataset
dataset_train = PneumoniaDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

# validation dataset
dataset_val = PneumoniaDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()

augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),
    
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ])
])

# Build model
model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])
    
%%time
LEARNING_RATE = 0.001

model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE*2,
            epochs=2,
            layers='heads',
            augmentation=None)  

history = model.keras_model.history.history    

%%time
model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE,
            epochs=6,
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

%%time
model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE/5,
            epochs=14,
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

%%time
model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE/50,
            epochs=20,
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

# loss
epochs = range(1,len(next(iter(history.values())))+1)
df_history = pd.DataFrame(history, index=epochs)

plt.figure(figsize=(17,5))

plt.subplot(131)
plt.plot(epochs, history["loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(132)
plt.plot(epochs, history["mrcnn_class_loss"], label="Train class ce")
plt.plot(epochs, history["val_mrcnn_class_loss"], label="Valid class ce")
plt.legend()
plt.subplot(133)
plt.plot(epochs, history["mrcnn_bbox_loss"], label="Train box loss")
plt.plot(epochs, history["val_mrcnn_bbox_loss"], label="Valid box loss")
plt.legend()
plt.show()
