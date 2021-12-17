import cv2 
import os
import random

from process_dataset import (_contrast_shift_pass, _brightness_shift_pass, _noise_pass, _blur_pass)

path_dataset = os.path.join("dataset","utkface")

img_filenames = os.listdir(path_dataset)
random_filename = random.choice(img_filenames)

path_img = os.path.join(path_dataset,random_filename)

orig_img = cv2.imread(path_img)

contr_img = _contrast_shift_pass(orig_img)
bright_img = _brightness_shift_pass(orig_img)
noise_img = _noise_pass(orig_img)
blur_img = _blur_pass(orig_img)

cv2.imshow("Original", orig_img)
cv2.imshow("Blurred", blur_img)
cv2.imshow("Noisy", noise_img)
cv2.imshow("Contrast Shifted", contr_img)
cv2.imshow("Brightness Shifted", bright_img)

cv2.waitKey(0)