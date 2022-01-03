import os 
import csv
import cv2

from tqdm import tqdm
from Dataset import Dataset

y_pred = {}
yp_path = os.path.join("trained_models", "pred_labels.csv")

val_dset = Dataset("validation","dataset")
val_dset.load("validation","val_label.csv")

with open(yp_path,"r", newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        y_pred[row[0]] = [int(row[1]),int(row[2]),int(row[3])]
c = 0
for k,l in tqdm(y_pred.items()):
    t_l = val_dset.labels[k]
    for v1,v2 in zip(t_l,l):
        if v1 != v2:
            break
    else:
        continue
    fn = str(l[0])+str(l[1])+str(l[2])+"-"+str(t_l[0])+str(t_l[1])+str(t_l[2])+"."+str(c)+".jpg"
    cv2.imwrite(os.path.join("trained_models","mispred",fn), val_dset.samples[k])
    c += 1

    

