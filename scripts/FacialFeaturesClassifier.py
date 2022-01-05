import pickle
import os
import numpy as np
import cv2
import dlib

from collections import OrderedDict
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from LBPDescriptor import LBPDescriptor
from HOGDescriptor import HOGDescriptor

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

detector = dlib.get_frontal_face_detector()
predictor_68_landm = dlib.shape_predictor( os.path.join('trained_models','shape_predictor_68_face_landmarks.dat'))
    
def _get_feature_landmarks(landms, name):
    si,ei = landm_ids_68[name]
    f_landm = np.array(landms[si:ei])
    return f_landm

def _check_empty(slice):
    return slice.shape[0] <= 0 or slice.shape[1] <= 0
    

def _shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def _cut_img(img):
    face_boxes = detector(img,1)
    landms = None
    c = 0
    for face in face_boxes:
        c+=1
        landms = _shape_to_np(predictor_68_landm(img,face))
        break
    cut = img.shape[0] // 2
    cut1 = cut + (cut // 2)
    if c == 0:
        up = img[:cut]
        mid = img[cut:cut1]
        low = img[cut1:]
    else:
        mouth_landm = _get_feature_landmarks(landms, "mouth")
        nose_landm = _get_feature_landmarks(landms, "nose")
        _,maxy_n = np.max(nose_landm,axis=0)
        _,miny_m = np.min(mouth_landm, axis=0)
        up = img[:maxy_n]
        mid = img[maxy_n:miny_m]
        low = img[miny_m:]

    if _check_empty(up) or _check_empty(mid) or _check_empty(low):
        up = img[:cut]
        mid = img[cut:cut1]
        low = img[cut1:]
        
    return up,mid,low

def _split_labels(labels):
    return (labels[:,0],labels[:,1],labels[:,2])

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
    

    def _cut_img(self, img):
        cut = int(img.shape[0] / 2)
        cut1 = img.shape[0] // 5
        up = img[:cut]
        mid = img[cut: cut + cut1]
        low = img[cut:]
        return up,low,mid

