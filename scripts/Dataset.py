import os 
import csv
import cv2
import random

import numpy as np
from tqdm import tqdm

class Dataset:
    def __init__(self, name, path_to_dataset = "", labels = None, samples = None):
        self.path = path_to_dataset
        self.name = name
        self.data_shape = None

        if labels is not None:
            self.labels = labels
        else:
            self.labels = {}
        if samples is not None:
            self.samples = samples
        else:
            self.samples = {}


    def __len__(self):
        return len(self.labels)

    def load_h5():
        pass

    def save_h5():
        pass

    def to_lists(self):
        ids = self.samples.keys()
        return [self.samples[k] for k in ids],np.array([self.labels[k] for k in ids],dtype='int32'),ids

    def augment(self, aug_processes, target):
        n = len(self.samples)
        if n == 0: 
            return
        perc = (float(target)/n) - 1.0
        c = 0

        new_samples = {}
        new_labels = {}

        if perc < 1.0:
            random_samples = random.sample(list(self.samples.items()),k= target - n)
            for fname,sample in random_samples:
                split_fname = fname.split(".")
                split_fname.insert(-1,str(c))
                gen_fname = '.'.join(split_fname)

                aug_proc = random.sample(aug_processes, k = 1)[0]

                gen_sample = aug_proc(sample)
                new_samples[gen_fname] = gen_sample
                new_labels[gen_fname] = self.labels[fname]
                c += 1
        else:
            floor_perc = int(np.floor(perc))
            diff = perc - floor_perc
            n_diff = diff * n
            replicas = 0
            i = 0
            for fname,sample in self.samples.items():
                replicas = floor_perc
                if i <= n_diff:
                    replicas +=1
                if replicas >= len(aug_processes):
                    aug_procs = random.choices(aug_processes, k = replicas)
                else:
                    aug_procs = random.sample(aug_processes, k = replicas)
                for j in range(replicas):
                    split_fname = fname.split(".")
                    split_fname.insert(-1,str(c))
                    gen_fname = '.'.join(split_fname)

                    aug_proc = aug_procs[j]

                    gen_sample = aug_proc(sample)
                    new_samples[gen_fname] = gen_sample
                    new_labels[gen_fname] = self.labels[fname]

                    c += 1
                    i += 1
        for k,v in new_samples.items():
            self.labels[k] = new_labels[k]
            self.samples[k] = v

    def reduce(self, target):
        n = len(self.samples)
        fnames = list(self.samples.keys())
        random_samples = random.sample(fnames,k=target)
        self.labels = { k : self.labels[k] for k in random_samples}
        self.samples =  { k : self.samples[k] for k in random_samples}
    
    def split_percs(self, perc : list, names : list):
        if len(perc) != len(names):
            return

        n = len(self)

        if n <= 1 or n < len(perc):
            return 

        splits = zip(names,perc)
        fnames = list(self.labels.keys())
        perm_fnames = random.sample(fnames, k = n)
        split_datasets = []
        start = 0
        for name,p in splits:
            split = int(np.ceil(n * p))
            split_fnames = perm_fnames[start:start+split]
            split_samples = {k : self.samples[k] for k in split_fnames}
            split_labels = {k : self.labels[k] for k in split_fnames}
            split_datasets.append(Dataset(name,path_to_dataset=self.path,samples=split_samples, labels=split_labels))
            start += split
        
        return split_datasets

    def split_cuts(self, cuts:list, names:list):
        if len(cuts) != len(names):
            return

        n = len(self)

        if n <= 1 or n < len(cuts):
            return

        splits = zip(cuts,names)
        fnames = list(self.labels.keys())
        perm_fnames = random.sample(fnames, k = n)
        split_datasets = []
        start = 0
        for c,name in splits:
            split_fnames = perm_fnames[start:start+c]
            split_samples = {k : self.samples[k] for k in split_fnames}
            split_labels = {k : self.labels[k] for k in split_fnames}
            split_datasets.append(Dataset(name,path_to_dataset=self.path,samples=split_samples, labels=split_labels))
            start += c
        return split_datasets


    def merge(self, other):
        for k,v in other.samples.items():
            self.samples[k] = v
        
        for k,v in other.labels.items():
            self.labels[k] = v

    def load(self, sample_folder, label_file, sample_limit = 0):
        self._read_from_folder(sample_folder, sample_limit)
        self._read_label_file(label_file)
        self.data_shape = next(iter(self.samples.values())).shape

    def save(self, sample_folder, label_file):
        self._save_to_folder(sample_folder)
        self._write_label_file(label_file)

    def _save_to_folder(self, folder_name):
        folder_path = os.path.join(self.path, folder_name)
        for k,v in tqdm(self.samples.items(), desc = "Saving samples of dataset %s" % (self.name)):
            cv2.imwrite(os.path.join(folder_path,k), v)

    def _read_from_folder(self, folder_name,sample_limit):
        folder_path = os.path.join(self.path, folder_name)
        filenames = os.listdir(folder_path)
        c = 0
        for fname in tqdm(filenames,desc="Loading samples for dataset %s" % (self.name)):
            if sample_limit > 0 and c >= sample_limit:
                break
            self.samples[fname] = cv2.imread(os.path.join(folder_path, fname)) 
            c += 1

    def _read_label_file(self, filename):
        print("Reading labels for dataset %s" % (self.name))
        with open(os.path.join(self.path,filename), "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.labels[row[0]] = [int(row[1]),int(row[2]),int(row[3])]

    def _write_label_file(self, filename):
        print("Writing labels for dataset %s" % (self.name))
        with open(os.path.join(self.path,filename), "w", newline = "") as csvfile:
            writer = csv.writer(csvfile)
            for k,v in self.labels.items():
                writer.writerow([k,v[0],v[1],v[2]])