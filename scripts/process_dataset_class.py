    
    
import os
import random
import cv2
import csv
import numpy as np

from tqdm import tqdm
from skimage.util import random_noise

def get_class_groups(label_dict):
    groups = [[],[],[],[]]
    flag = 0
    for k,v in label_dict.items():
        flag = 0
        for i in range(3):
            if v[i] == 1:
                flag = 1
                groups[i].append(k)
        if flag == 0:
            groups[3].append(k)
    return groups


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

def _blur_pass(img, sigmaX = None):
    sx = 0
    if sigmaX is not None:
        sx = sigmaX
    return cv2.GaussianBlur(img, (3,3), sx)

def _noise_pass(img):
    float_img = random_noise(img, var= random.randrange(1,11) * 0.002)
    return np.array(255*float_img, dtype = 'uint8')

def _brightness_shift_pass(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    val = 0
    rand = random.randint(-80,80)
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

def _contrast_shift_pass(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=random.uniform(0.3,4), tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def get_aug_processes():
    return  [_blur_pass, _noise_pass, _brightness_shift_pass, _contrast_shift_pass]

def augmentation(group_imgs, group_filenames,labels,target_number):
    if len(group_imgs) == 0: 
        return group_imgs, group_filenames, labels
    group_size = len(group_imgs)
    perc = (target_number/len(group_imgs)) - 1.0
    aug_processes = get_aug_processes()

    gen_img = None
    gen_filename = None
    
    c = 0
    if perc < 1.0:
        random_indexes = random.sample(range(0,group_size),k= target_number - group_size)
        for i in random_indexes:
            split_filename = group_filenames[i].split(".")
            split_filename.insert(-1,str(c))
            gen_filename = '.'.join(split_filename)

            aug_proc = random.sample(aug_processes, k = 1)[0]

            gen_img = aug_proc(group_imgs[i])
            group_imgs.append(gen_img)
            group_filenames.append(gen_filename)

            try:
                labels[gen_filename] = labels[group_filenames[i]]
            except KeyError as e:
                print(e)

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
                
                gen_img = aug_proc(group_imgs[i])
                group_imgs.append(gen_img)
                group_filenames.append(gen_filename)                
                labels[gen_filename] = labels[group_filenames[i]]

                c += 1
    return group_imgs,group_filenames, labels

def save_data_on_disk(folder_path,filenames,imgs,labels):
    for i in range(len(imgs)):
        filename = filenames[i]
        filepath = os.path.join(folder_path,filename)
        label = labels[filename]
        if os.path.exists(filepath):
            new_filename = filename.split(".")
            new_filename.insert(-2, str(i))
            filename = ".".join(new_filename)
            filepath = os.path.join(folder_path, filename)
        cv2.imwrite(filepath, aug_imgs[i])
        writer.writerow([filename,label[0],label[1],label[2]])

if __name__ == "__main__":
    path_dataset = "dataset"
    path_images = os.path.join(path_dataset,"utkface")
    path_train = os.path.join(path_dataset,"train")
    path_validation = os.path.join(path_dataset, "validation")
    path_labels = os.path.join(path_dataset,"labels.csv")

    dataset_filenames = os.listdir(path_images)

    
    #Processing parameters
    validation_split = 0.15
    n_per_class = 1000
    #max_augmentation = 5

    #Dictionaries for storing samples and labels
    label_dict = {}
    img_dict = {}

    with open(path_labels,"r") as csvfile:
        reader = csv.reader(csvfile, delimiter = ",")
        for row in reader:
            label_dict[row[0]] = (int(row[1]),int(row[2]),int(row[3]))

    
    for filename in tqdm(dataset_filenames, desc = "Loading images"):
        img_dict[filename] = preprocessing(cv2.imread(os.path.join(path_images,filename)))
    

    class_groups = get_class_groups(label_dict)

    try:
        os.mkdir(path_train)
        os.mkdir(path_validation)
    except OSError as e:
        pass

    with open(os.path.join(path_dataset, "train_label.csv"),"w", newline = '') as train_label_file:
        with open(os.path.join(path_dataset, "val_label.csv"), "w", newline = '') as val_label_file:
            writer = csv.writer(train_label_file)
            val_writer = csv.writer(val_label_file)    
            for i in tqdm(range(len(class_groups))):
                group = class_groups[i]
                n = len(group)
                if i == 3:
                    n_per_class *= 3

                #Splitting into training and validation
                validation_filenames = []
                
                train_filenames = []
                train_imgs = []
                train_labels = {}

                if n > 1:
                    k =  min(n, n_per_class)
                    perm_fnames = random.sample(group, k = k)
                    split = int(np.ceil(k * validation_split))
                    validation_filenames = perm_fnames[:split]
                    validation_imgs = [img_dict[val_fname] for val_fname in validation_filenames]

                    train_filenames = perm_fnames[split:]
                    train_imgs = [img_dict[train_fname] for train_fname in train_filenames]
                    for train_fname in train_filenames:
                        try:
                            train_labels[train_fname] = label_dict[train_fname]
                        except KeyError:
                            pass                    
                    save_data_on_disk(path_validation, validation_filenames, validation_imgs, label_dict)
                else:
                    train_filenames.append(group[0])
                    train_imgs.append(img_dict[train_filenames[0]])
                    train_labels[train_filenames[0]] = label_dict[train_filenames[0]]
              
                train_size = len(train_filenames)
                target = n_per_class
                
                if train_size < target:
                    aug_imgs, aug_filenames, aug_labels = augmentation(train_imgs, train_filenames, train_labels, target)
                    save_data_on_disk(path_train, aug_filenames, aug_imgs, aug_labels)
                elif train_size > target:
                    random_indexes = random.sample(range(0,n),k=n_per_class)
                    us_train_filenames = [train_filenames[i] for i in random_indexes]
                    us_train_imgs = [train_imgs[i] for i in random_indexes]
                    save_data_on_disk(path_train,us_train_filenames,us_train_imgs,label_dict)