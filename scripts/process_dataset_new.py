    
    
import os
import random
import cv2
import csv
import numpy as np

from Dataset import Dataset
from tqdm import tqdm
from skimage.util import random_noise

def get_class_groups(label_dict):
    groups = [[],[],[],[]]
    flag = 0
    for k,v in label_dict.items():
        flag = 0
        for i in range(3):
            if v[i] == 1:
                flag = 1
                groups[i].append(k)
        if flag == 0:
            groups[3].append(k)
    return groups


def count_classes(labels):
    counts = [0,0,0]
    positive_counts = [0,0,0]
    for _,v in labels.items():
        if v[0] == '1':
            positive_counts[0] += 1
        else:
            counts[0] += 1
        if v[1] == '1':
            positive_counts[1] += 1
        else:
            counts[1] += 1
        if v[2] == '1':
            positive_counts[2] += 1
        else:
            counts[2] += 1
    counts.extend(positive_counts)

    return counts

def preprocessing(img):
    return img

def _blur_pass(img, sigmaX = None):
    sx = 0
    if sigmaX is not None:
        sx = sigmaX
    return cv2.GaussianBlur(img, (3,3), sx)

def _noise_pass(img):
    float_img = random_noise(img, var= random.randrange(1,11) * 0.002)
    return np.array(255*float_img, dtype = 'uint8')

def _brightness_shift_pass(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    val = 0
    rand = random.randint(-80,80)
    for x in range(v.shape[0]):
        for y in range(v.shape[1]):
            val = v[x][y]
            if rand >= 0:
                v[x][y] = min(255, val + rand)
            else:
                v[x][y] = max(0, val + rand)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def _contrast_shift_pass(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=random.uniform(0.3,4), tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def get_aug_processes():
    return  [_blur_pass, _noise_pass, _brightness_shift_pass, _contrast_shift_pass]

def split_by_labels(dset):
    class_dsets = []
    class_dsets.append(Dataset("class1",dset.path))
    class_dsets.append(Dataset("class2",dset.path))
    class_dsets.append(Dataset("class3",dset.path))
    class_dsets.append(Dataset("noclass",dset.path))

    for k,v in dset.labels.items():
        flag = 0
        for i in range(3):
            if v[i] == 1:
                flag = 1
                class_dsets[i].labels[k] = v
                class_dsets[i].samples[k] = dset.samples[k]
        if flag == 0:
            class_dsets[3].labels[k] = v
            class_dsets[3].samples[k] = dset.samples[k]
    return class_dsets

if __name__ == "__main__":
    path_dataset = "dataset"
    path_images = os.path.join(path_dataset,"utkface")
    path_train = os.path.join(path_dataset,"train")
    path_validation = os.path.join(path_dataset, "validation")
    path_labels = os.path.join(path_dataset,"labels.csv")

    dataset_filenames = os.listdir(path_images)
    try:
        os.mkdir(path_train)
        os.mkdir(path_validation)
    except OSError as e:
        pass
    
    #Processing parameters
    validation_split = 0.15
    train_per_class = 1000
    val_per_class = int(train_per_class * validation_split)

    full_dset = Dataset("full", path_dataset)
    full_dset.load("utkface", "labels.csv")

    #class_groups = get_class_groups(dset.labels)
    class_dsets = split_by_labels(full_dset)
    train_dset = Dataset("train", full_dset.path)
    val_dset = Dataset("validation", full_dset.path)

    for d in tqdm(class_dsets):
        size = d.size()
        if size > train_per_class + val_per_class:
            d.reduce(train_per_class + val_per_class)
            train,val = d.split_percs([1.0-validation_split,validation_split], ["train", "validation"])
        else:
            train,val = d.split_percs([1.0-validation_split,validation_split], ["train", "validation"])
            train.augment(get_aug_processes(),train_per_class)
        train_dset.merge(train)
        val_dset.merge(val)
    train_dset.save("train", "train_label.csv")
    val_dset.save("validation", "val_label.csv")