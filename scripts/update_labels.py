import os
import csv

from Dataset import Dataset

PRIME = 1009

path_noise = os.path.join("dataset", "testset_noisy")
path_smorfie = os.path.join("dataset", "testset_smorfie")
path_tilted = os.path.join("dataset", "testset_tilted")
path_default = os.path.join("dataset", "testset_default")

noise_filenames = os.listdir(path_noise)
smorfie_filenames = os.listdir(path_smorfie)
tilted_filenames = os.listdir(path_tilted)
default_filenames = os.listdir(path_default)

csv_files = ['test_set1.csv','test_set2.csv','test_set3.csv','test_set4.csv','test_set5.csv']
dest_csv_file = ['noisy_labels.csv','smorfie_labels.csv','default_labels.csv','tilted_labels.csv']
testset_filenames = [noise_filenames,smorfie_filenames,default_filenames,tilted_filenames]
all_labels = {}

for csv_f in csv_files: 
    with open(os.path.join('dataset',csv_f),'r') as file:
        reader = csv.reader(file)
        for row in reader:
            all_labels[row[0]] = (int(row[1]),int(row[2]),int(row[3]))

for csv_f,fnames in zip(dest_csv_file, testset_filenames):
    with open(os.path.join('dataset',csv_f),'w', newline='\n') as file:
        writer = csv.writer(file)
        for fname in fnames:
            l = all_labels[fname]
            writer.writerow([fname, l[0], l[1], l[2]])