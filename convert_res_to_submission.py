import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob



def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]

    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

    return bboxes


if __name__ == "__main__":
    # test_dir = '/media/data/VinBigData/VinBigData3xDownsample/test/test'
    test_meta_csv = '/home/yujia/Desktop/kaggle/yolov5/test_meta.csv'
    test_df = pd.read_csv(test_meta_csv)
    image_ids = []
    PredictionStrings = []

    for file_path in tqdm(glob('/home/yujia/Desktop/kaggle/yolov5/runs/detect/orig_f1_1024_c01_i4_c1_26/labels/*txt')):
        image_id = file_path.split('/')[-1].split('.')[0]
        w, h = test_df.loc[test_df.image_id == image_id, ['width', 'height']].values[0]
        f = open(file_path, 'r')
        data = np.array(f.read().replace('\n', ' ').strip().split(' ')).astype(np.float32).reshape(-1, 6)
        data = data[:, [0, 5, 1, 2, 3, 4]]
        bboxes = list(np.round(np.concatenate((data[:, :2], np.round(yolo2voc(h, w, data[:, 2:]))), axis=1).reshape(-1),
                               5).astype(str))
        for idx in range(len(bboxes)):
            bboxes[idx] = str(int(float(bboxes[idx]))) if idx % 6 != 1 else bboxes[idx]
        image_ids.append(image_id)
        PredictionStrings.append(' '.join(bboxes))

    pred_df = pd.DataFrame({'image_id': image_ids,
                            'PredictionString': PredictionStrings})
    sub_df = pd.merge(test_df, pred_df, on='image_id', how='left').fillna("14 1 0 0 1 1")
    sub_df = sub_df[['image_id', 'PredictionString']]
    num_neg = len(sub_df[sub_df['PredictionString'] == "14 1 0 0 1 1"])
    print(f"neg predictions: {num_neg}")
    sub_df.to_csv('/home/yujia/Desktop/kaggle/Detection/orig_f1_1024_c01_i4_c1_26.csv', index=False)
    sub_df.tail()
