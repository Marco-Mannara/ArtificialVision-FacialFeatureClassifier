import cv2 
import os
import random

from Dataset import Dataset

#from process_dataset import (_contrast_shift_pass, _brightness_shift_pass, _noise_pass, _blur_pass)

dset = Dataset("orig", "dataset")
dset.load("utkface","labels.csv")

class1_labels = [(k,v) for k,v in dset.labels.items() if v[0] == 1]
class2_labels = [(k,v) for k,v in dset.labels.items() if v[1] == 1]
class3_labels = [(k,v) for k,v in dset.labels.items() if v[2] == 1]

class1_sample = (class1_labels[0][1], dset.samples[class1_labels[0][0]])
class2_sample = (class2_labels[0][1], dset.samples[class2_labels[0][0]])
class3_sample = (class3_labels[0][1], dset.samples[class3_labels[0][0]])


#contr_img = _contrast_shift_pass(orig_img)
#bright_img = _brightness_shift_pass(orig_img)
#noise_img = _noise_pass(orig_img)
#blur_img = _blur_pass(orig_img)
#cut = int(orig_img.shape[0] / 2)
#cut_img1 = orig_img[:cut]
#cut_img2 = orig_img[cut:]

#cv2.imshow("Original", orig_img)
cv2.imshow("class 1", class1_sample[1])
cv2.imshow("class 2", class2_sample[1])
cv2.imshow("class 3", class3_sample[1])
#cv2.imshow("Blurred", blur_img)
#cv2.imshow("Noisy", noise_img)
#cv2.imshow("Contrast Shifted", contr_img)
#cv2.imshow("Brightness Shifted", bright_img)

cv2.waitKey(0)