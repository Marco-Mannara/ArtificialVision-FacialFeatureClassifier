import os
import cv2
import csv
import h5py
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    train_path = os.path.join("dataset", "train")
    val_path = os.path.join("dataset", "validation")
    
    train_filenames = os.listdir(train_path)
    val_filenames = os.listdir(val_path)
    train_labels_dict = {}
    val_labels_dict = {}

    with open(os.path.join("dataset","train_label.csv")) as train_csvfile:
        reader = csv.reader(train_csvfile)
        for row in tqdm(reader):
            train_labels_dict[row[0]] = (int(row[1]),int(row[2]),int(row[3]))

    with open(os.path.join("dataset","val_label.csv")) as val_csvfile:
        reader = csv.reader(val_csvfile)
        for row in tqdm(reader):
            val_labels_dict[row[0]] = (int(row[1]),int(row[2]),int(row[3]))

    train_imgs = np.zeros(shape=(len(train_filenames),224,224,3),dtype="float32")
    val_imgs = np.zeros(shape=(len(val_filenames),224,224,3),dtype="float32")
    train_labels = np.zeros(shape=(len(train_filenames), 6),dtype="float32")
    val_labels = np.zeros(shape=(len(val_filenames), 6),dtype="float32")

    for i in tqdm(range(len(train_filenames))):
        train_imgs[i][...] = (cv2.resize(cv2.imread(os.path.join(train_path,train_filenames[i])), (224,224)) / 255.0)[...]
        fname_meta = train_filenames[i].split(".")[0].split("_")
        label = None
        for j in range(3):
            label = train_labels_dict[train_filenames[i]]
            train_labels[i][j] = label[j]
        train_labels[i][3] = int(fname_meta[0])
        train_labels[i][4] = int(fname_meta[1])
        try:
            train_labels[i][5] = int(fname_meta[2])
        except OverflowError as err:
            train_labels[i][5] = 0

    for i in tqdm(range(len(val_filenames))):
        val_imgs[i][...] = (cv2.resize(cv2.imread(os.path.join(val_path,val_filenames[i])), (224,224)) / 255.0)[...]
        fname_meta = val_filenames[i].split(".")[0].split("_")
        label = None
        for j in range(3):
            label = val_labels_dict[val_filenames[i]]
            val_labels[i][j] = label[j]
        val_labels[i][3] = int(fname_meta[0])
        val_labels[i][4] = int(fname_meta[1])
        try:
            val_labels[i][5] = int(fname_meta[2])
        except OverflowError as err:
            val_labels[i][5] = 0
 

    with h5py.File(os.path.join("dataset","dataset"),'w') as data_file:
        data_file.create_dataset("train/samples", data=train_imgs)
        data_file.create_dataset("train/labels", data=train_labels)
        data_file.create_dataset("val/samples", data=val_imgs)
        data_file.create_dataset("val/labels", data=val_labels)
        