from threading import local
import seaborn as sns
import random 
import numpy as np
import matplotlib as mlp
import cv2

from Dataset import Dataset
from LBPDescriptor import LBPDescriptor
from skimage.feature import local_binary_pattern


num_points = 8
radius = 2

dset = Dataset("dset", "dataset")
dset.load("utkface", "labels.csv", 10)
rand = None
for k in dset.samples.keys():
    rand = k
    break
sample = (rand, dset.labels[rand], cv2.cvtColor(dset.samples[rand], cv2.COLOR_BGR2GRAY))

descriptor = LBPDescriptor(num_points,radius,16)

hist,lbp_img = descriptor.describe(sample[2])
lbp_img /= 2**num_points
#hist = np.array(hist, dtype="float32")
sns.set_theme()
tips = sns.load_dataset("tips")
plot = sns.displot(tips)

cv2.imshow("original - gray", sample[2])
cv2.imshow("lbp", lbp_img)

cv2.waitKey(0)
