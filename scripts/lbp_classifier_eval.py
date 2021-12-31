import pickle 
import os 
import numpy as np

from Dataset import Dataset
from FacialFeaturesClassifier import FFClassifier

from sklearn.metrics import accuracy_score,multilabel_confusion_matrix

if __name__ == "__main__":
    eval_dset = Dataset("evaluation", "dataset")
    eval_dset.load("validation", "val_label.csv")

    classifier = FFClassifier()
    classifier.load()
    X,y_t = eval_dset.to_lists()
    y_p = classifier.predict(X)

    accuracy = accuracy_score(y_t,y_p)
    conf_matrix = multilabel_confusion_matrix(y_t,y_p)
    
    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix:\n", conf_matrix)

    print("Done.")