import os
import csv

from Dataset import Dataset
from FacialFeaturesClassifier import FFClassifier

from sklearn.metrics import accuracy_score,balanced_accuracy_score,multilabel_confusion_matrix

if __name__ == "__main__":
    MODEL_NAME = "ffc_model.pickle"


    eval_dset = Dataset("evaluation", "dataset")
    eval_dset.load("validation", "val_label.csv")

    classifier = FFClassifier()
    classifier.load(name = MODEL_NAME)
    X,y_t,fnames = eval_dset.to_lists()
    y_p = classifier.predict(X)

    accuracy = accuracy_score(y_t,y_p)
    conf_matrix = multilabel_confusion_matrix(y_t,y_p)
    
    with open(os.path.join("trained_models", "pred_labels.csv"), "w", newline='') as file:
        writer = csv.writer(file)
        for fname, l in zip(fnames,y_p):
            writer.writerow([fname,l[0],l[1],l[2]])

    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix:\n", conf_matrix)

    print("Done.")