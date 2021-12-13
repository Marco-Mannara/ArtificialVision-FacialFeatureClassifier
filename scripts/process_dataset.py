import os
import random
import cv2
import csv
import numpy as np
import pickle 

def count_age_groups(filenames):
    age_counts = {}
    for item in filenames:
        age = int(item.split("_")[0])
        if age not in age_counts.keys():
            age_counts[age] = 1
        else:
            age_counts[age] += 1

    return sorted(age_counts.items())

def preprocessing(img):
    return img

def augmentation(group_imgs, group_filenames,target_number):
    factor = target_number/len(group_imgs)
    perc = factor - 1.0

    if perc <= 1.0:
        pass
    else:
        pass



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
            aug_imgs,aug_filenames = augmentation(group_imgs,group_filenames,target)
            for i in range(len(group_imgs)):
                cv2.imwrite(os.path.join(path_train,aug_filenames[i]), aug_imgs[i])
        elif n > n_per_age:
            random_indexes = random.sample(range(0,n),k=n_per_age)
            for i in random_indexes:
                cv2.imwrite(os.path.join(path_train, group_filenames[i]), group_imgs[i])


    






