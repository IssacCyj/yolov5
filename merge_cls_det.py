import pandas as pd
import numpy as np
from glob import glob
import shutil

low_thr  = 0.01
high_thr = 0.90
def filter_2cls(row, low_thr=low_thr, high_thr=high_thr):
    prob = row['average']
    if prob<low_thr:
        ## Less chance of having any disease
        row['PredictionString'] = '14 1 0 0 1 1'
    elif low_thr<=prob<high_thr:
        ## More change of having any diesease
        row['PredictionString']+=f' 14 {prob} 0 0 1 1'
    elif high_thr<=prob:
        ## Good chance of having any disease so believe in object detection model
        row['PredictionString'] = row['PredictionString']
    else:
        raise ValueError('Prediction must be from [0-1]')
    return row


pred_14cls = pd.read_csv('/home/yujia/Desktop/kaggle/Detection/60_01_4_f1_aug.csv')
pred_2cls = pd.read_csv('/home/yujia/Desktop/kaggle/VINXChest/Classification/classification.csv')


if __name__ == "__main__":
    pred = pd.merge(pred_14cls, pred_2cls, on='image_id', how='left')

    sub = pred.apply(filter_2cls, axis=1)
    print(sub[sub['PredictionString']=='14 1 0 0 1 1'].shape[0])
    sub[['image_id', 'PredictionString']].to_csv('/home/yujia/Desktop/kaggle/Detection/0190_60_01_4_f1_aug.csv', index=False)


