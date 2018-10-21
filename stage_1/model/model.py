import os
import sys
import random
import pandas as pd
from tqdm import tqdm
import pydicom
import numpy as np

from pneumonia import PneumoniaDataset, PneumoniaConfig, InferenceConfig
from functions import get_image_fps, box_locations, iou, create_submission

DATA_DIR = '../input/kaggle/'
TRAIN_DIR = os.path.join(DATA_DIR, 'stage_1_train_images')
TEST_DIR = os.path.join(DATA_DIR, 'stage_1_test_images')

MODEL_DIR = '../model/Mask_RCNN'
ORIG_SIZE = 1024

# Import Mask RCNN
sys.path.append(os.path.join(MODEL_DIR))  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log

inference_config = InferenceConfig()
inference_config.display()
# trained model 
dir_names = next(os.walk(MODEL_DIR))[1]
key = inference_config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)

if not dir_names:
    import errno
    raise FileNotFoundError(
        errno.ENOENT,
        "Could not find model directory under {}".format(self.model_dir))

fps = []

for d in dir_names: 
    dir_name = os.path.join(MODEL_DIR, d)
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        print('No weight files in {}'.format(dir_name))
    else:
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        fps.append(checkpoint)

model_path = sorted(fps)[-1]
print('model_path=', model_path)

# Load trained model
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Load trained weights 
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Load validation file paths
image_fps_val = pd.read_csv('image_fps_val.csv').image_fps_val.tolist()
print('validation_size=', len(image_fps_val))

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
        
        if len(r['rois'])==0:
            prediction[patient_id]=[]
        else:
            prediction[patient_id]=[]
            
            for i in range(len(r['rois'])):
                if r['scores'][i] > min_conf:
                    score = r['scores'][i]
                    x1 = r['rois'][i][1]
                    y1 = r['rois'][i][0]
                    width = r['rois'][i][3] - x1
                    height = r['rois'][i][2] - y1
                    
                    x1*=RESIZE_FACTOR
                    y1*=RESIZE_FACTOR
                    width*=RESIZE_FACTOR
                    height*=RESIZE_FACTOR
                    
                    prediction[patient_id].append([score, x1, y1, width, height])
                
    return prediction
    
prediction = predict(image_fps_val)
truth = box_locations()
iou_all_mean,tp,fp,tn,fn = iou(truth, prediction)
print(iou_all_mean,tp,fp,tn,fn)

# Predict on testing data and create submission
if True:
    image_fps_test = get_image_fps(TEST_DIR)
    image_fps_test.sort()

    prediction_test = predict(image_fps_test, min_conf=0.94, augment=False)
    
    create_submission(prediction_test)
    
submission = pd.read_csv('submission.csv')
submission.sort_values(by='patientId').head(20)
