import os
import sys
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from skimage import measure
from skimage.transform import resize
sys.path.append(os.path.join('../model'))

import importlib

import unet
import resnet
import functions
import generator

importlib.reload(unet)
importlib.reload(resnet)
importlib.reload(functions)
importlib.reload(generator)
from unet import UNet
from resnet import ResNet
from generator import Generator
from functions import iou_segmentation, iou_box, LearningRate

# load bounding box locations
box_locations = {}

temp = pd.read_csv(os.path.join('../input/kaggle/stage_1_train_labels.csv'))
box_locations = defaultdict(list)

print('num of patients=', len(temp.patientId.unique()))
print('num of boxes=', len(temp))

for _,row in temp.iterrows():
    
    if row.Target==1:
        patientId=row.patientId
        box=[int(row.x), int(row.y), int(row.width), int(row.height)]
        box_locations[patientId].append(box)
        
box_locations['640743e2-59ed-4859-b25b-eb92f351cef0']

# split training and validation
train_df = '../input/kaggle/stage_1_train_images'
filenames = os.listdir(train_df)
random.shuffle(filenames)
n_validation = 2560
train_filenames = filenames[n_validation:]
validation_filenames = filenames[:n_validation]
print('number of train samples=', len(train_filenames))
print('number of validation samples=', len(validation_filenames))

BATCH_SIZE = 16
IMAGE_SIZE = 320
N_EPOCH = 5

# build model
model = UNet(input_shape=(IMAGE_SIZE,IMAGE_SIZE,1))
model.compile(optimizer='adam',
              loss=keras.losses.binary_crossentropy,
              metrics=[iou_segmentation])

# visualize model
plot_model(UNet, 'model.png', show_shapes=True)
model.summary()

callbacks=[]
callback_lr = LearningRate(lr=0.0001)
callbacks = [callback_lr]

# training
train_df = os.path.join('../input/kaggle/stage_1_train_images')
train_gen = Generator(folder, 
                      train_filenames, 
                      box_locations, 
                      batch_size=BATCH_SIZE, 
                      image_size=IMAGE_SIZE, 
                      shuffle=True, 
                      augment=True, 
                      predict=False)

valid_gen = Generator(folder, 
                      validation_filenames, 
                      box_locations, 
                      batch_size=BATCH_SIZE, 
                      image_size=IMAGE_SIZE, 
                      shuffle=False, 
                      augment=False,
                      predict=False)

history = model.fit_generator(train_gen, validation_data=valid_gen, callbacks=callbacks, epochs=N_EPOCH, shuffle=True)

# training history
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(history.epoch, history.history["loss"], label="Train loss")
plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(122)
plt.plot(history.epoch, history.history["iou_segmentation"], label="Train iou")
plt.plot(history.epoch, history.history["val_iou_segmentation"], label="Valid iou")
plt.legend()
plt.show()

# predict validation images
prob_thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
nthresh = len(prob_thresholds)

# load validation filenames
folder = os.path.join('../data/stage_1_train_images')
test_filenames = validation_filenames
print('number of test samples=', len(test_filenames))

# create test generator with predict flag set to True
test_gen = Generator(folder, test_filenames, None, batch_size=25, 
                     image_size=IMAGE_SIZE, shuffle=False, predict=True)

count = 0
ns = nthresh*[0]
nfps = nthresh*[0]
ntps = nthresh*[0]
overall_maps = nthresh*[0.]
for imgs, filenames in test_gen:
    # predict batch of images
    preds = model.predict(imgs)
    # loop through batch
    for pred, filename in zip(preds, filenames):
        count = count + 1
        maxpred = np.max(pred)
        # resize predicted mask
        pred = resize(pred, (1024, 1024), mode='reflect')
        # threshold predicted mask
        boxes_preds = []
        scoress = []
        for thresh in prob_thresholds:
            comp = pred[:, :, 0] > thresh
            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            boxes_pred = np.empty((0,4),int)
            scores = np.empty((0))
            for region in measure.regionprops(comp):
                y, x, y2, x2 = region.bbox
                boxes_pred = np.append(boxes_pred, [[x, y, x2-x, y2-y]], axis=0)
                conf = np.mean(pred[y:y2, x:x2])
                scores = np.append( scores, conf )
            boxes_preds = boxes_preds + [boxes_pred]
            scoress = scoress + [scores]
        boxes_true = np.empty((0,4),int)
        fn = filename.split('.')[0]
        
        # if image contains pneumonia
        if fn in box_locations:
            # loop through pneumonia
            for location in box_locations[fn]:
                x, y, w, h = location
                boxes_true = np.append(boxes_true, [[x, y, w, h]], axis=0)
        for i in range(nthresh):
            if (boxes_true.shape[0]==0 and boxes_preds[i].shape[0]>0):
                ns[i] = ns[i] + 1
                nfps[i] = nfps[i] + 1
            elif (boxes_true.shape[0]>0):
                ns[i] = ns[i] + 1
                contrib = iou_box(boxes_true, boxes_preds[i], scoress[i]) 
                overall_maps[i] = overall_maps[i] + contrib
                if (boxes_preds[i].shape[0]>0):
                    ntps[i] = ntps[i] + 1

    if count >= len(test_filenames):
        break

for i, thresh in enumerate(prob_thresholds):
    print("\nProbability threshold=", thresh)
    overall_maps[i] = overall_maps[i] / (ns[i]+1e-7)
    print("False positive cases=", nfps[i])
    print("True positive cases=", ntps[i])
    print("Overall evaluation score=", overall_maps[i])

# save model
model_path = 'model'+datetime.now().strftime("%Y%m%d_%H:%M:%S")+'.hdf5'
model.save(os.path.join(model_path))
