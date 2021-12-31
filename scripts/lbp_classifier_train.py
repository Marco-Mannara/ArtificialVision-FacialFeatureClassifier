import os

from collections import OrderedDict
from Dataset import Dataset
from FacialFeaturesClassifier import FFClassifier

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

if __name__ == "__main__":
	path_dataset = "dataset"
	train_dset = Dataset("train", path_dataset) 

	train_dset.load("train", "train_label.csv")

	try:
		os.mkdir("trained_models")
	except OSError:
		pass
	
	classifier = FFClassifier(verbose=True)
	X,y = train_dset.to_lists()
	classifier.fit(X,y)
	classifier.save()