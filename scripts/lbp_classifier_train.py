from colorsys import yiq_to_rgb
import dlib
import cv2
import os
import pickle

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.svm import LinearSVC

from Dataset import Dataset
from FacialFeaturesClassifier import FFClassifier
from LocalBinaryPatterns import LocalBinaryPatterns

landm_ids_68 = OrderedDict([
("mouth", (48, 68)),
("inner_mouth", (60, 68)),
("right_eyebrow", (17, 22)),
("left_eyebrow", (22, 27)),
("right_eye", (36, 42)),
("left_eye", (42, 48)),
("nose", (27, 36)),
("jaw", (0, 17))
])

def prepare_data(dset):
	desc = LocalBinaryPatterns(12, 5)
	dset_sample_ids = list(dset.labels.keys())
	data = []
	gray_imgs = np.zeros(shape = (len(dset_sample_ids), dset.data_shape[0],dset.data_shape[1]), dtype='uint8')
	imgs,labels = (np.array([dset.samples[k] for k in dset_sample_ids]),np.array([dset.labels[k] for k in dset_sample_ids]))
	class1_labels = labels[:,0]
	class2_labels = labels[:,1]
	class3_labels = labels[:,2]
	i = 0
	for img in tqdm(imgs, desc="Calculating LBP %s dataset images" % (dset.name)):
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		gray_imgs[i][...] = gray[...]
		hist = desc.describe(gray)
		data.append(hist)
		i+=1

	return (data, class1_labels, class2_labels, class3_labels, gray_imgs)


if __name__ == "__main__":
	path_dataset = "dataset"
	train_dset = Dataset("train", path_dataset) 

	train_dset.load("train", "train_label.csv")

	try:
		os.mkdir("trained_models")
	except OSError:
		pass
	
	classifier = LBPClassifier(verbose=True)
	X,y = train_dset.to_lists()
	classifier.fit(X,y)
	classifier.save()


	'''
	train_data, c1_train_labels, c2_train_labels, c3_train_labels,gray_imgs = prepare_data(train_dset)

	print("Training first classifier.")
	model1 = LinearSVC(C=1.0, random_state=42)
	model1.fit(train_data, c1_train_labels)
	print("Training second classifier.")
	model2 = LinearSVC(C=1.0, random_state=42)
	model2.fit(train_data, c2_train_labels)
	print("Training third classifier.")
	model3 = LinearSVC(C=1.0, random_state=42)
	model3.fit(train_data, c3_train_labels)

	with open(os.path.join("trained_models","lbp_model.pickle"), "wb") as file:
		pickle.dump({"model1":model1, "model2":model2, "model3":model3}, file)'''