import os
import random
import cv2
import csv
import numpy as np
import pickle

from tqdm import tqdm
from skimage.util import random_noise

def count_age_groups(filenames):
    age_counts = {}
    for item in filenames:
        age = int(item.split("_")[0])
        if age not in age_counts.keys():
            age_counts[age] = 1
        else:
            age_counts[age] += 1

    return sorted(age_counts.items())

def count_classes(labels):
    counts = [0,0,0]
    positive_counts = [0,0,0]
    for _,v in labels.items():
        if v[0] == '1':
            positive_counts[0] += 1
        else:
            counts[0] += 1
        if v[1] == '1':
            positive_counts[1] += 1
        else:
            counts[1] += 1
        if v[2] == '1':
            positive_counts[2] += 1
        else:
            counts[2] += 1
    counts.extend(positive_counts)

    return counts


def preprocessing(img):
    return img

def _blur_pass(img):
    return cv2.GaussianBlur(img, (3,3), 0)

def _noise_pass(img):
    float_img = random_noise(img)
    return np.array(255*float_img, dtype = 'uint8')

def _brightness_shift_pass(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    val = 0
    rand = random.randint(-40,40)
    for x in range(v.shape[0]):
        for y in range(v.shape[1]):
            val = v[x][y]
            if rand >= 0:
                v[x][y] = min(255, val + rand)
            else:
                v[x][y] = max(0, val + rand)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def contrast_shift_pass(img):   
    return img
'''
def add_googles_pass():
    pass

def add_beard_pass():
    pass
'''
def get_aug_processes():
    return  [_blur_pass, _noise_pass, _brightness_shift_pass, contrast_shift_pass]

def augmentation(group_imgs, group_filenames,labels,target_number):
    group_size = len(group_imgs)
    perc = (target_number/len(group_imgs)) - 1.0
    aug_processes = get_aug_processes()

    gen_img = None
    gen_filename = None
    
    c = 0
    #d = {}
    if perc < 1.0:
        random_indexes = random.sample(range(0,group_size),k= target_number - group_size)
        for i in random_indexes:
            split_filename = group_filenames[i].split(".")
            split_filename.insert(-1,str(c))
            gen_filename = '.'.join(split_filename)

            aug_proc = random.sample(aug_processes, k = 1)[0]
            #d[gen_filename] = aug_proc.__name__
            gen_img = aug_proc(group_imgs[i])
            group_imgs.append(gen_img)
            group_filenames.append(gen_filename)
            labels[gen_filename] = labels[group_filenames[i]]

            c += 1
    else:
        floor_perc = int(np.floor(perc))
        diff = perc - floor_perc
        n_diff = diff * group_size
        replicas = 0

        for i in range(group_size):
            replicas = floor_perc
            if i <= n_diff:
                replicas +=1
            if replicas >= len(aug_processes):
                aug_procs = random.choices(aug_processes, k = replicas)
            else:
                aug_procs = random.sample(aug_processes, k = replicas)
            for j in range(replicas):
                split_filename = group_filenames[i].split(".")
                split_filename.insert(-1,str(c))
                gen_filename = '.'.join(split_filename)

                aug_proc = aug_procs[j]
                #d[gen_filename] = aug_proc.__name__
                gen_img = aug_proc(group_imgs[i])
                group_imgs.append(gen_img)
                group_filenames.append(gen_filename)
                labels[gen_filename] = labels[group_filenames[i]]

                c += 1


   
        


        

    return group_imgs,group_filenames, labels





if __name__ == "__main__":
    path_dataset = "dataset"
    path_images = os.path.join(path_dataset,"utkface")
    path_train = os.path.join(path_dataset,"train")
    path_labels = os.path.join(path_dataset,"labels.csv")

    dataset_filenames = os.listdir(path_images)
    age_counts = count_age_groups(dataset_filenames)

    label_dict = {}

    with open(path_labels,"r") as csvfile:
        reader = csv.reader(csvfile, delimiter = ",")
        for row in reader:
            label_dict[row[0]] = tuple(row[1:])

    try:
        os.mkdir(path_train)
    except OSError as e:
        pass

    n_per_age = 250
    max_augmentation = 5
    
    dataset = []
    for count in age_counts:
        age,n = count

        group_filenames = [x for x in dataset_filenames if int(x.split("_")[0]) == age]
        group_imgs = []
        for filename in group_filenames:
            img = cv2.imread(os.path.join(path_images,filename))
            img = preprocessing(img)
            group_imgs.append(img)

        if n < n_per_age:
            target = n_per_age
            if n_per_age / n > max_augmentation:
                target = n * max_augmentation
            aug_imgs, aug_filenames, aug_labels = augmentation(group_imgs,group_filenames, label_dict, target)
            for i in range(len(aug_imgs)):
                cv2.imwrite(os.path.join(path_train,aug_filenames[i]), aug_imgs[i])
        elif n > n_per_age:
            random_indexes = random.sample(range(0,n),k=n_per_age)
            for i in random_indexes:
                cv2.imwrite(os.path.join(path_train,group_filenames[i]), group_imgs[i])
       


    






