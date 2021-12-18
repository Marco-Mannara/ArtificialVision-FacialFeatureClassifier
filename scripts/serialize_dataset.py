import os
import cv2
import csv
import pickle
from tqdm import tqdm



if __name__ == "__main__":
    train_path = os.path.join("dataset", "train")
    val_path = os.path.join("dataset", "validation")
    
    train_filenames = os.listdir(train_path)
    val_filenames = os.listdir(val_path)

    train_imgs = {}
    train_labels = {}
    val_imgs = {}
    val_labels = {}

    for fname in tqdm(train_filenames):
        train_imgs[fname] = cv2.imread(os.path.join(train_path,fname))

    for fname in tqdm(val_filenames):
        val_imgs[fname] = cv2.imread(os.path.join(val_path,fname))
    
    with open(os.path.join("dataset","train_label.csv")) as train_csvfile:
        reader = csv.reader(train_csvfile)
        for row in tqdm(reader):
            train_labels[row[0]] = (int(row[1]),int(row[2]),int(row[3]))

    with open(os.path.join("dataset","val_label.csv")) as val_csvfile:
        reader = csv.reader(val_csvfile)
        for row in tqdm(reader):
            val_labels[row[0]] = (int(row[1]),int(row[2]),int(row[3]))

    with open(os.path.join("dataset","train_data"), "wb") as train_file:
        pickle.dump({"samples":train_imgs, "labels": train_labels}, train_file)

    with open(os.path.join("dataset","val_data"), "wb") as val_file:
        pickle.dump({"samples":val_imgs, "labels": val_labels}, val_file)