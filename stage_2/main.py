!pip3 install -q pydicom 
!pip3 install -q tqdm 
!pip3 install -q imgaug

import os, sys, glob, copy
import random
import math
import numpy as np
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm

import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log

from pneumonia import InferenceConfig, InferenceConfig2
from utils import get_image_fps, box_locations, iou, create_submission, testing_augment

DATA_DIR = '../data'
TRAIN_DIR = os.path.join(DATA_DIR, 'stage_1_train_images')
TEST_DIR = os.path.join(DATA_DIR, 'stage_2_test_images')

MODEL_DIR = '../model/Mask_RCNN'
ORIG_SIZE = 1024

sys.path.append(os.path.join(MODEL_DIR))  

# Load validation file paths
image_fps_val = pd.read_csv('../stage_1/image_fps_val.csv').image_fps_val.tolist()
print('validation_size=', len(image_fps_val))

# Stage 1 config
inference_config = InferenceConfig()
inference_config.display()
assert inference_config.NUM_CLASSES == 2

model_path = '../model/Mask_RCNN/pneumonia20181018T1640/mask_rcnn_pneumonia_0020.h5'

# Load stage 1 trained model
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Predict Stage 1
def predict(image_fps, min_conf=0.95, augment=False):
    RESIZE_FACTOR = ORIG_SIZE / inference_config.IMAGE_SHAPE[0]
    prediction={}
    
    for image_id in tqdm(image_fps):
        ds = pydicom.read_file(image_id)
        image = ds.pixel_array
        
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=inference_config.IMAGE_MIN_DIM,
            min_scale=inference_config.IMAGE_MIN_SCALE,
            max_dim=inference_config.IMAGE_MAX_DIM,
            mode=inference_config.IMAGE_RESIZE_MODE)

        patient_id = os.path.splitext(os.path.basename(image_id))[0]

        r = model.detect([image])
        r = r[0]
        
        if augment:
            r2 = model.detect([np.fliplr(image)])
            r2 = r2[0]
            
            r = testing_augment(r, r2, min_conf, inference_config)        

        if len(r['rois'])==0:
            prediction[patient_id]=[]
        else:
            prediction[patient_id]=[]
            
            for i in range(len(r['rois'])):
                if r['scores'][i] > min_conf:
                    score = r['scores'][i]
                    x = r['rois'][i][1]
                    y = r['rois'][i][0]
                    
                    if x>0 and y>0:
                        width = r['rois'][i][3] - x
                        height = r['rois'][i][2] - y

                        x*=RESIZE_FACTOR
                        y*=RESIZE_FACTOR
                        width*=RESIZE_FACTOR
                        height*=RESIZE_FACTOR
                    
                        prediction[patient_id].append([score, x, y, width, height])
                
    return prediction
    
    
truth = box_locations()
prediction = predict(image_fps_val, min_conf=0.96, augment=True)

iou_all_mean,tp,fp,tn,fn = iou(truth, prediction)
print(iou_all_mean,tp,fp,tn,fn)

# Predict on all training data for training stage 2 model
if False:
    image_fps_train = get_image_fps(TRAIN_DIR)
    prediction = predict(image_fps_train, min_conf=0.96, augment=True)
    
    # Convert prediction to training labels
    train_labels_2 = pd.DataFrame(columns=['patientId', 'x', 'y', 'width', 'height', 'Target', 'class'])
    i=0
    for patient_id in list(prediction.keys()):

        if len(truth[patient_id])>0:
            for box in truth[patient_id]:
                train_labels_2.loc[i] = [patient_id, int(box[0]), int(box[1]), int(box[2]), int(box[3]), 1, 1]
                i+=1
        else:
            if len(prediction[patient_id])>0:
                for box in prediction[patient_id]:
                    train_labels_2.loc[i] = [patient_id, int(box[1]), int(box[2]), int(box[3]), int(box[4]), 0, 2]
                    i+=1
            else:
                train_labels_2.loc[i] = [patient_id, np.nan, np.nan, np.nan, np.nan, 0, 0]
                i+=1

    train_labels_2.sort_values(by='patientId', inplace=True)
    train_labels_2.to_csv(os.path.join(DATA_DIR, 'train_labels_2.csv'), index=False)
    print(len(train_labels_2))
    train_labels_2.head()
    
# Config for Stage 2
inference_config_2 = InferenceConfig2()
inference_config_2.display()
assert inference_config_2.NUM_CLASSES == 3

# Load stage 2 model
model_2_path = '../model/Mask_RCNN/pneumonia20181021T0214/mask_rcnn_pneumonia_0020.h5'
model_2 = modellib.MaskRCNN(mode='inference', 
                          config=inference_config_2,
                          model_dir=MODEL_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_2_path != "", "Provide path to trained weights"
print("Loading weights from ", model_2_path)
model_2.load_weights(model_2_path, by_name=True)


# Stage 2 prediction
def predict2(image_fps, min_conf=0.90, augment=False):
    RESIZE_FACTOR = ORIG_SIZE / inference_config_2.IMAGE_SHAPE[0]
    prediction={}
    
    for image_id in tqdm(image_fps):
        ds = pydicom.read_file(image_id)
        image = ds.pixel_array
        
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=inference_config.IMAGE_MIN_DIM,
            min_scale=inference_config.IMAGE_MIN_SCALE,
            max_dim=inference_config.IMAGE_MAX_DIM,
            mode=inference_config.IMAGE_RESIZE_MODE)

        patient_id = os.path.splitext(os.path.basename(image_id))[0]
                
        r = model_2.detect([image])
        r = r[0]
        
        if augment:
            r2 = model_2.detect([np.fliplr(image)])
            r2 = r2[0]
            
            #print(r2['scores'])
            
            r = testing_augment(r, r2, min_conf, inference_config)
        

        if len(r['rois'])==0:
            prediction[patient_id]=[]
        else:
            prediction[patient_id]=[]
            
            for i in range(len(r['rois'])):
                if r['class_ids'][i]==2 and r['scores'][i] > min_conf:
                    score = r['scores'][i]
                    x = r['rois'][i][1]
                    y = r['rois'][i][0]
                    
                    if x>0 and y>0:
                        width = r['rois'][i][3] - x
                        height = r['rois'][i][2] - y

                        x*=RESIZE_FACTOR
                        y*=RESIZE_FACTOR
                        width*=RESIZE_FACTOR
                        height*=RESIZE_FACTOR
                    
                        prediction[patient_id].append([score, x, y, width, height])
                
    return prediction


prediction_2 = predict2(image_fps_val, min_conf=0.92, augment=False)

def merge_predictions(prediction, prediction_2):
    prediction_3 = copy.deepcopy(prediction)
    
    for patient_id in list(prediction_2.keys()):
        if len(prediction_2[patient_id])>0:
            prediction_3[patient_id] = []
    
    return prediction_3
    
prediction_3 = merge_predictions(prediction, prediction_2)

iou_all_mean,tp,fp,tn,fn = iou(truth, prediction_3)
print(iou_all_mean,tp,fp,tn,fn)  # 0.21805178140096618 248 235 948 69

# Prepare prediction set on test data
if True:
    image_fps_test = get_image_fps(TEST_DIR)
    image_fps_test.sort()

    prediction_test = predict(image_fps_test, min_conf=0.96, augment=True)
    prediction_test_2 = predict2(image_fps_test, min_conf=0.92, augment=False)
    prediction_test_3 = merge_predictions(prediction_test, prediction_test_2)
    
    create_submission(prediction_test_3)
    
submission = pd.read_csv('prediction.csv')
