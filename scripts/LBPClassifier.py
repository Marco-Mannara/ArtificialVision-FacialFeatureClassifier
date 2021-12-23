import pickle
import os

from sklearn.svm import LinearSVC

class LBPClassifier:
    def __init__(self):
        self.model1 = LinearSVC(C=1.0, random_state=42)
        self.model2 = LinearSVC(C=1.0, random_state=42)
        self.model3 = LinearSVC(C=1.0, random_state=42)

    def predict(self):
        pass

    def fit(self, data, labels):
        pass

    def save(self, path):
        with open(os.path.join("trained_models","lbp_model.pickle"), "wb") as file:
            pickle.dump({"model1":self.model1, "model2":self.model2, "model3":self.model2},file)

    def load(self, path):
        with open(os.path.join("trained_models","lbp_model.pickle"), "rb") as file:
            models = pickle.load(file)
        self.model1 = models["model1"]
        self.model2 = models["model2"]
        self.model3 = models["model3"]


