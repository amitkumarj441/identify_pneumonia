import tensorflow as tf
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import glob


def iou_box(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at different intersection over union (IoU) threshold. Only used after testing.
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection 
        
    # return None if both are empty, don't count the image in final evaluation
    
    def iou(box1, box2):
        x11, y11, w1, h1 = box1
        x21, y21, w2, h2 = box2
        assert w1 * h1 > 0
        assert w2 * h2 > 0
        x12, y12 = x11 + w1, y11 + h1
        x22, y22 = x21 + w2, y21 + h2

        area1, area2 = w1 * h1, w2 * h2
        xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        else:
            intersect = (xi2-xi1) * (yi2-yi1)
            union = area1 + area2 - intersect
            return intersect / union
    
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    
    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    
    map_total = 0
    
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m
    
    return map_total / len(thresholds)


def iou(truth, prediction, patient_id=None):
    """
    IoU for the entire validation dataset
    
    input:
        truth: dictionary, patient_id as key, list of bounding boxes [[x,y,width,height]] as value
        prediction: dictionary, patient_id as key, list of bounding boxes [[score,x,y,width,height]] as value
    
    output:
        iou,tp,fp,tn,fn
        
    """
    all_iou = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    if patient_id==None:   
        for patient_id in list(prediction.keys()):

            if len(truth[patient_id])==0 and len(prediction[patient_id])==0:
                tn += 1
            elif len(truth[patient_id])==0 and len(prediction[patient_id])>0:
                fp += 1
                all_iou.append(0)
            elif len(truth[patient_id])>0 and len(prediction[patient_id])==0:
                fn += 1
                all_iou.append(0)
            elif len(truth[patient_id])>0 and len(prediction[patient_id])>0:
                tp +=1
                
                box_truth = truth[patient_id]               
                box_prediction = [[box[1], box[2], box[3], box[4]] for box in prediction[patient_id]]
                scores = [box[0] for box in prediction[patient_id]]
                all_iou.append(iou_box(np.array(box_truth), np.array(box_prediction), scores))
                
        return np.array(all_iou).mean(), tp, fp, tn, fn
    else:
        pass

    
def box_locations():
    box_locations = {}

    temp = pd.read_csv(os.path.join('../input/kaggle/stage_1_train_labels.csv'))
    box_locations = defaultdict(list)
    
    for _,row in temp.iterrows():

        if row.Target==1:
            patientId=row.patientId
            box=[int(row.x), int(row.y), int(row.width), int(row.height)]
            box_locations[patientId].append(box)
    
    return box_locations


def create_submission(prediction, output='submission.csv'):

    with open(output, 'w') as file:
        file.write("patientId,PredictionString\n")

        for patient_id, boxes in prediction.items():
            
            out_str = ""
            out_str += patient_id
            out_str += ","

            if len(boxes) == 0:
                pass
            else:
                for box in boxes:
                    out_str += ' '
                    out_str += str(round(box[0], 2))
                    out_str += ' '

                    bboxes_str = "{} {} {} {}".format(box[1], box[2], box[3], box[4])
                    out_str += bboxes_str

            file.write(out_str+"\n")

            
def get_image_fps(image_dir):
    image_fps = glob.glob(image_dir+'/'+'*.dcm')
    
    return list(set(image_fps))


def parse_dataset(image_dir, anns): 
    image_fps = get_image_fps(image_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(image_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    
    return image_fps, image_annotations 
