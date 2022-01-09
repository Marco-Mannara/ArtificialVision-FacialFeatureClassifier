import os

from collections import OrderedDict
from Dataset import Dataset
from FacialFeaturesClassifier import FFClassifier
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

if __name__ == "__main__":
	MODEL_NAME = "ffc_model_noalign.pickle"

	path_dataset = "dataset"
	train_dset = Dataset("train", path_dataset)  

	train_dset.load("train3", "train_label3.csv")

	try:
		os.mkdir("trained_models")
	except OSError:
		pass
	
	classifier = FFClassifier(verbose=True)
	X,y,_ = train_dset.to_lists()
	classifier.fit(X,y)
	classifier.save(name=MODEL_NAME)

	y_pred = classifier.predict(X)
	accuracy = accuracy_score(y,y_pred)
	conf_matrix = multilabel_confusion_matrix(y,y_pred)

	print("Accuracy:", round(accuracy, 4))
	print("Confusion Matrix:\n", conf_matrix)