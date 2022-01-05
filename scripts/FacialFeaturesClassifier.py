import pickle
import os
import numpy as np
import cv2

from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from LBPDescriptor import LBPDescriptor
from HOGDescriptor import HOGDescriptor

class FFClassifier:
    def __init__(self, verbose = False):
        self.model1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=200)
        self.model2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=400)
        self.model3 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=200)
        '''
        self.model1 = LinearSVC(C=10.0, random_state=41, max_iter=1000000)
        self.model2 = LinearSVC(C=10.0, random_state=42, max_iter=1000000)
        self.model3 = LinearSVC(C=10.0, random_state=43, max_iter=1000000)
        '''
        self.verbose = verbose

    def predict(self,samples):
        d1,d2,d3 = self._prepare_data(samples)

        pred1 = self.model1.predict(d1)
        pred2 = self.model2.predict(d2)
        pred3 = self.model3.predict(d3)

        return [[pred1[i],pred2[i],pred3[i]] for i in range(len(pred1))]

    def fit(self, samples, labels):
        d1,d2,d3 = self._prepare_data(samples)
        l1,l2,l3 = self._split_labels(labels)
        if self.verbose:
            print("Training first model.")                           
        self.model1.fit(d1,l1)
        if self.verbose:
            print("Training second model.")                           
        self.model2.fit(d2,l2)
        if self.verbose:
            print("Training third model.")                           
        self.model3.fit(d3,l3)

    def save(self, path = "trained_models", name = "lbp_model.pickle"):
        with open(os.path.join(path,name), "wb") as file:
            pickle.dump({"model1":self.model1, "model2":self.model2, "model3":self.model3},file)

    def load(self, path = "trained_models", name = "lbp_model.pickle"):
        with open(os.path.join(path,name), "rb") as file:
            models = pickle.load(file)
        self.model1 = models["model1"]
        self.model2 = models["model2"]
        self.model3 = models["model3"]

    def _prepare_data(self, samples):
        if self.verbose:
            print("Preparing data...")
        lbp_desc = LBPDescriptor(6, 1, 24, 4, 4)
        #lbp_desc2 = LBPDescriptor(9, 1, 16, 4, 4)
        #hog_desc = HOGDescriptor(8,(32,32),(2,2))
        #hog_desc2 = HOGDescriptor(8,(16,16),(3,3))
        data1,data2,data3 = ([],[],[])
        for img in tqdm(samples, desc="Calculating LBP for images"):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            up_half,low_half, mid = self._cut_img(gray)
            hist1,_ = lbp_desc.describe(low_half)
            data1.append(hist1)
            hist2,_ = lbp_desc.describe(mid)
            data2.append(hist2)
            hist3,_ = lbp_desc.describe(up_half)
            data3.append(hist3)

        return (data1, data2, data3)
    
    def _split_labels(self, labels):
        return (labels[:,0],labels[:,1],labels[:,2])

    def _cut_img(self, img):
        cut = int(img.shape[0] / 2)
        cut1 = img.shape[0] // 5
        up = img[:cut]
        mid = img[cut: cut + cut1]
        low = img[cut:]
        return up,low,mid

