import os 
import csv
import cv2

import numpy as np
from tqdm import tqdm

class Dataset:
    def __init__(self, name, path_to_dataset = "",):
        self.path = path_to_dataset
        self.name = name
        #self.sub_groups = sub_groups

        self.labels = {}
        self.samples = {}
    
    def load_h5():
        pass

    def save_h5():
        pass

    def augment(self, augment_procs):
        pass

    def split(self, splits_perc : list, split_names : list):
        pass

    def load(self, sample_folder, label_file):
        self._read_from_folder(sample_folder)
        self._read_label_file(label_file)

    def save(self, sample_folder, label_file):
        self._save_to_folder(sample_folder)
        self._write_label_file(label_file)

    def _save_to_folder(self, folder_name):
        folder_path = os.path.join(self.path, folder_name)
        for k,v in self.samples:
            cv2.imwrite(os.path.join(folder_path,k), v)

    def _read_from_folder(self, folder_name):
        folder_path = os.path.join(self.path, folder_name)
        filenames = os.listdir(folder_path)

        for fname in filenames:
            self.samples[fname] = cv2.imread(os.path.join(folder_path, fname)) 

    def _read_label_file(self, filename):
        with open(os.path.join(self.path,filename), "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.labels[row[0]] = (int(row[1]),int(row[2]),int(row[3]))

    def _write_label_file(self, filename):
        with open(os.path.join(self.path,filename), "w", newline = "") as csvfile:
            writer = csv.writer(csvfile)
            for k,v in self.labels.items():
                writer.writerow([k,v[0],v[1],v[2]])




