import os
import csv

from sklearn.metrics import accuracy_score, multilabel_confusion_matrix,balanced_accuracy_score
from FacialFeaturesClassifier import FFClassifier
from Dataset import Dataset


MODEL_NAME = "ffc_model_noalign"
DATASET_NAMES=['default','noisy','smorfie','tilted']
DATASET_FOLDERS = ['testset_default','testset_noisy','testset_smorfie','testset_tilted']
DATASET_LABELS = ['default_labels.csv','noisy_labels.csv','smorfie_labels.csv','tilted_labels.csv']
classifier = FFClassifier()
classifier.load(name = MODEL_NAME + ".pickle")

curr_dset = None

try:
    os.mkdir(os.path.join('trained_models',MODEL_NAME))
except OSError:
    pass

with open(os.path.join('trained_models',MODEL_NAME,'metrics.csv'),'w',newline = '') as file:
    m_writer = csv.writer(file)
    m_writer.writerow([
        'Test Set'
        'Beard Accuracy',
        'Beard Balanced Accuracy',
        'Moustache Accuracy',
        'Moustache Balanced Accuracy',
        'Glasses Accuracy',
        'Glasses Balanced Accuracy',
        'Global Average Accuracy',
        'Global Balanced Average Accuracy',
        'FAS'
    ])
    for name,folder,label in zip(DATASET_NAMES,DATASET_FOLDERS,DATASET_LABELS):
        curr_dset = Dataset(name,path_to_dataset='dataset')
        curr_dset.load(folder,label)
        X,y_t,fnames = curr_dset.to_lists()
        y_p = classifier.predict(X)

        with open(os.path.join('trained_models',MODEL_NAME,name + '_res.csv'),'w',newline='') as file:
            writer = csv.writer(file)
            for fname,p_l in zip(fnames,y_p):
                writer.writerow([fname, p_l[0],p_l[1],p_l[2]])

        b_gt = y_t[:,0]
        m_gt = y_t[:,1]
        g_gt = y_t[:,2]

        b_res = y_p[:,0]
        m_res = y_p[:,1]
        g_res = y_p[:,2]
        print("--Metrics for test set %s--" % (name))
        conf_matrix = multilabel_confusion_matrix(y_t,y_p)
        beard_accuracy = round(accuracy_score(b_gt, b_res),4)
        beard_balanced_accuracy = round(balanced_accuracy_score(b_gt, b_res),4)
        moustache_accuracy = round(accuracy_score(m_gt, m_res),4)
        moustache_balanced_accuracy = round(balanced_accuracy_score(m_gt, m_res),4)
        glasses_accuracy = round(accuracy_score(g_gt, g_res),4)
        glasses_balanced_accuracy = round(balanced_accuracy_score(g_gt, g_res),4)
        avg_accuracy = round((beard_accuracy + moustache_accuracy + glasses_accuracy) / 3,4)
        avg_balanced_accuracy = round((beard_balanced_accuracy + moustache_balanced_accuracy + glasses_balanced_accuracy) / 3,4)
        fas = round(avg_accuracy + avg_balanced_accuracy,4)
        print("Beard Accuracy:", beard_accuracy)
        print("Beard Balanced Accuracy:", beard_balanced_accuracy)
        print("Moustache Accuracy:", moustache_accuracy)
        print("Moustache Balanced Accuracy:", moustache_balanced_accuracy)
        print("Glasses Accuracy:", glasses_accuracy)
        print("Glasses Balanced Accuracy:", glasses_balanced_accuracy)
        print("Global Average Accuracy:", avg_accuracy)
        print("Global Balanced Average Accuracy:",avg_balanced_accuracy)
        print("FAS:", fas)
        print("Confusion Matrix:\n", conf_matrix)

        m_writer.writerow([
            name,
            beard_accuracy,
            beard_balanced_accuracy,
            moustache_accuracy,
            moustache_balanced_accuracy,
            glasses_accuracy,
            glasses_balanced_accuracy,
            avg_accuracy,
            avg_balanced_accuracy,
            fas
        ])