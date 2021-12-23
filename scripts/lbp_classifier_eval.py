import pickle 
import os 

import numpy as np

from lbp_classifier_train import prepare_data
from Dataset import Dataset
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    eval_dset = Dataset("evaluation", "dataset")
    eval_dset.load("validation", "val_label.csv")

    val_data, c1_val_label, c2_val_label, c3_val_label = prepare_data(eval_dset)
    with open(os.path.join("trained_models","lbp_model.pickle"), "rb") as file:
        models=pickle.load(file)

    model1 = models["model1"]
    model2 = models["model2"]
    model3 = models["model3"]

    m1_pred = model1.decision_function(val_data)
    m2_pred = model2.decision_function(val_data)
    m3_pred = model3.decision_function(val_data)

    m1_accuracy = round(accuracy_score(c1_val_label,m1_pred))
    m2_accuracy = round(accuracy_score(c2_val_label,m2_pred))
    m3_accuracy = round(accuracy_score(c3_val_label,m3_pred))
    print("m1_acc:", m1_accuracy, "m2_acc:", m2_accuracy, "m3_acc:", m3_accuracy)


