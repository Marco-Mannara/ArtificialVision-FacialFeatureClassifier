import pickle
import os
import numpy as np
import cv2

from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from LBPDescriptor import LBPDescriptor
from FaceAligner import FaceAligner

face_aligner = FaceAligner(desiredLeftEye=(0.37,0.28),desiredFaceWidth=200)

def _cut_img(img):
    cut = int(img.shape[0] / 2)
    cut1 = img.shape[0] // 5
    up = img[:cut]
    mid = img[cut: cut + cut1]
    low = img[cut:]
    return up,low,mid

def _split_labels(labels):
    return (labels[:,0],labels[:,1],labels[:,2])

def _preprocessing(img):
    _,aligned_gray,align_succ = face_aligner.align(img)
    gray = cv2.equalizeHist(aligned_gray)
    return gray,align_succ

class FFClassifier:
    def __init__(self, verbose = False):
        self.model1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=200)
        self.model2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=400)
        self.model3 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=200)
        self.verbose = verbose

    def predict(self,samples):
        d1,d2,d3 = self._prepare_data(samples)

        pred1 = self.model1.predict(d1)
        pred2 = self.model2.predict(d2)
        pred3 = self.model3.predict(d3)

        return np.array([[pred1[i],pred2[i],pred3[i]] for i in range(len(pred1))], dtype='int32')

    def fit(self, samples, labels):
        d1,d2,d3 = self._prepare_data(samples)
        l1,l2,l3 = _split_labels(labels)
        if self.verbose:
            print("Training first model.")                           
        self.model1.fit(d1,l1)
        if self.verbose:
            print("Training second model.")                           
        self.model2.fit(d2,l2)
        if self.verbose:
            print("Training third model.")                           
        self.model3.fit(d3,l3)

    def save(self, path = "trained_models", name = "ffc_model.pickle"):
        with open(os.path.join(path,name), "wb") as file:
            pickle.dump({"model1":self.model1, "model2":self.model2, "model3":self.model3},file)

    def load(self, path = "trained_models", name = "ffc_model.pickle"):
        with open(os.path.join(path,name), "rb") as file:
            models = pickle.load(file)
        self.model1 = models["model1"]
        self.model2 = models["model2"]
        self.model3 = models["model3"]

    def _prepare_data(self, samples):
        if self.verbose:
            print("Preparing data...")
        lbp_desc = LBPDescriptor(6, 1, 24, 1, 1)
        data1,data2,data3 = ([],[],[])

        for img in tqdm(samples, desc="Calculating features for images"):
            gray,_ = _preprocessing(img)
            #up_half,low_half, mid = _cut_img(img)
            hist1,_ = lbp_desc.describe(gray)
            data1.append(hist1)
            '''
            hist2,_ = lbp_desc.describe(mid)
            data2.append(hist2)
            hist3,_ = lbp_desc.describe(up_half)
            data3.append(hist3)
            '''
        return (data1, data1, data1)
    



