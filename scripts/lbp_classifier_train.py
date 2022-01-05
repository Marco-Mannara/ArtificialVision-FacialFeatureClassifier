import os

from collections import OrderedDict
from Dataset import Dataset
from FacialFeaturesClassifier import FFClassifier

if __name__ == "__main__":
	path_dataset = "dataset"
	train_dset = Dataset("train", path_dataset)  

	train_dset.load("train0", "train_label0.csv")

	try:
		os.mkdir("trained_models")
	except OSError:
		pass
	
	classifier = FFClassifier(verbose=True)
	X,y = train_dset.to_lists()
	classifier.fit(X,y)
	classifier.save()
	