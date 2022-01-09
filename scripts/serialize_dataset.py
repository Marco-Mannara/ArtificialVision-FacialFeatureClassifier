import os
import cv2
import csv
import h5py
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    suffix = "2"
    train_path = os.path.join("dataset", "train" + suffix)
    val_path = os.path.join("dataset", "validation" + suffix)
    trainl_path = os.path.join("dataset", "train_label" + suffix + ".csv")
    vall_path = os.path.join("dataset", "val_label" + suffix + ".csv")

    train_filenames = os.listdir(train_path)
    val_filenames = os.listdir(val_path)
    train_labels_dict = {}
    val_labels_dict = {}

    with open(trainl_path) as train_csvfile:
        reader = csv.reader(train_csvfile)
        for row in tqdm(reader):
            train_labels_dict[row[0]] = (int(row[1]),int(row[2]),int(row[3]))

    with open(vall_path) as val_csvfile:
        reader = csv.reader(val_csvfile)
        for row in tqdm(reader):
            val_labels_dict[row[0]] = (int(row[1]),int(row[2]),int(row[3]))

    train_imgs = np.zeros(shape=(len(train_filenames),200,200,3),dtype="uint8")
    val_imgs = np.zeros(shape=(len(val_filenames),200,200,3),dtype="uint8")
    train_labels = np.zeros(shape=(len(train_filenames), 3),dtype="uint8")
    val_labels = np.zeros(shape=(len(val_filenames), 3),dtype="uint8")

    for i in tqdm(range(len(train_filenames))):
        train_imgs[i][...] = cv2.imread(os.path.join(train_path,train_filenames[i]))[...]
        fname_meta = train_filenames[i].split(".")[0].split("_")
        label = None
        for j in range(3):
            label = train_labels_dict[train_filenames[i]]
            train_labels[i][j] = label[j]

    for i in tqdm(range(len(val_filenames))):
        val_imgs[i][...] = cv2.imread(os.path.join(val_path,val_filenames[i]))[...]
        fname_meta = val_filenames[i].split(".")[0].split("_")
        label = None
        for j in range(3):
            label = val_labels_dict[val_filenames[i]]
            val_labels[i][j] = label[j]
 

    with h5py.File(os.path.join("dataset","dataset"+suffix),'w') as data_file:
        data_file.create_dataset("train/samples", data=train_imgs)
        data_file.create_dataset("train/labels", data=train_labels)
        data_file.create_dataset("val/samples", data=val_imgs)
        data_file.create_dataset("val/labels", data=val_labels)
        