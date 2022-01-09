import os
import csv

from sklearn.metrics import accuracy_score, multilabel_confusion_matrix,balanced_accuracy_score
from FacialFeaturesClassifier import FFClassifier
from Dataset import Dataset

MODEL_NAME = "ffc_model_nocuts_nogrid.pickle"
DATASET_NAMES=['default','noisy','smorfie','tilted']
DATASET_FOLDERS = ['testset_default','testset_noisy','testset_smorfie','testset_tilted']
DATASET_LABELS = ['default_labels.csv','noisy_labels.csv','smorfie_labels.csv','tilted_labels.csv']
classifier = FFClassifier()
classifier.load(name = MODEL_NAME)
curr_dset = None
for name,folder,label in zip(DATASET_NAMES,DATASET_FOLDERS,DATASET_LABELS):
    curr_dset = Dataset(name,path_to_dataset='dataset')
    curr_dset.load(folder,label)
    X,y_t,fnames = curr_dset.to_lists()
    y_p = classifier.predict(X)

    b_gt = y_t[:,0]
    m_gt = y_t[:,1]
    g_gt = y_t[:,2]

    b_res = y_p[:,0]
    m_res = y_p[:,1]
    g_res = y_p[:,2]
    print("--Metrics for test set %s--" % (name))
    conf_matrix = multilabel_confusion_matrix(y_t,y_p)
    beard_accuracy = accuracy_score(b_gt, b_res)
    beard_balanced_accuracy = balanced_accuracy_score(b_gt, b_res)
    moustache_accuracy = accuracy_score(m_gt, m_res)
    moustache_balanced_accuracy = balanced_accuracy_score(m_gt, m_res)
    glasses_accuracy = accuracy_score(g_gt, g_res)
    glasses_balanced_accuracy = balanced_accuracy_score(g_gt, g_res)
    avg_accuracy = (beard_accuracy + moustache_accuracy + glasses_accuracy) / 3
    avg_balanced_accuracy = (beard_balanced_accuracy + moustache_balanced_accuracy + glasses_balanced_accuracy) / 3
    fas = avg_accuracy + avg_balanced_accuracy
    print("Beard Accuracy:", round(beard_accuracy,4))
    print("Beard Balanced Accuracy:", round(beard_balanced_accuracy,4))
    print("Moustache Accuracy:", round(moustache_accuracy,4))
    print("Moustache Balanced Accuracy:", round(moustache_balanced_accuracy,4))
    print("Glasses Accuracy:", round(glasses_accuracy,4))
    print("Glasses Balanced Accuracy:", round(glasses_balanced_accuracy,4))
    print("Global Average Accuracy:", round(avg_accuracy,4))
    print("Global Balanced Average Accuracy:",round(avg_balanced_accuracy,4))
    print("FAS:", round(fas,4))
    print("Confusion Matrix:\n", conf_matrix)