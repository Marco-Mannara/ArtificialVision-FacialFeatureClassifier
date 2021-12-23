import os
import random
import cv2
import csv
import numpy as np

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


if __name__ == "__main__":
    path_dataset = "dataset"
    path_images = os.path.join(path_dataset,"utkface")
    path_train = os.path.join(path_dataset,"train")
    path_validation = os.path.join(path_dataset, "validation")
    path_labels = os.path.join(path_dataset,"labels.csv")

    dataset_filenames = os.listdir(path_images)
    age_counts = count_age_groups(dataset_filenames)
    
    #Processing parameters
    validation_split = 0.15
    n_per_age = 250
    max_augmentation = 5

    #Dictionaries for storing samples and labels
    label_dict = {}
    img_dict = {}

    with open(path_labels,"r") as csvfile:
        reader = csv.reader(csvfile, delimiter = ",")
        for row in reader:
            label_dict[row[0]] = tuple(row[1:])

    for filename in tqdm(dataset_filenames, desc = "Loading images"):
        img_dict[filename] = preprocessing(cv2.imread(os.path.join(path_images,filename)))


    try:
        os.mkdir(path_train)
        os.mkdir(path_validation)
    except OSError as e:
        pass

    with open(os.path.join(path_dataset, "train_label.csv"),"w", newline = '') as train_label_file:
        with open(os.path.join(path_dataset, "val_label.csv"), "w", newline = '') as val_label_file:
            writer = csv.writer(train_label_file)
            val_writer = csv.writer(val_label_file)    
          
            for count in tqdm(age_counts, desc="Processing Dataset"):
                age,n = count
                group_filenames = [x for x in dataset_filenames if int(x.split("_")[0]) == age]

                #Splitting into training and validation
                validation_filenames = []
                
                train_filenames = []
                train_imgs = []
                train_labels = {}

                if n > 1:
                    k =  min(n, n_per_age)
                    perm_fnames = random.sample(group_filenames, k = k)
                    split = int(np.ceil(k * validation_split))
                    validation_filenames = perm_fnames[:split]
                    train_filenames = perm_fnames[split:]
                    train_imgs = [img_dict[train_fname] for train_fname in train_filenames]
                    for train_fname in train_filenames:
                        try:
                            train_labels[train_fname] = label_dict[train_fname]
                        except KeyError:
                            pass
                else:
                    train_filenames.append(group_filenames[0])
                    train_imgs.append(img_dict[train_filenames[0]])
                    train_labels[train_filenames[0]] = label_dict[train_filenames[0]]

                #Writing validation set to disk
                for val_fname in validation_filenames:
                    label = label_dict[val_fname]
                    val_writer.writerow([val_fname,label[0],label[1],label[2]])
                    cv2.imwrite(os.path.join(path_validation, val_fname),img_dict[val_fname])
                
                
                train_size = len(train_filenames)
                if train_size < 1:
                    print("There's a problem.")

                target = n_per_age
                if n_per_age / train_size > max_augmentation:
                    target = train_size * max_augmentation
         
                if train_size < target:
                    aug_imgs, aug_filenames, aug_labels = augmentation(train_imgs, train_filenames, train_labels, target)
                    for i in range(len(aug_imgs)):
                        cv2.imwrite(os.path.join(path_train,aug_filenames[i]), aug_imgs[i])
                    for k,v in aug_labels.items():
                        writer.writerow([k,v[0],v[1],v[2]])
                elif train_size > target:
                    random_indexes = random.sample(range(0,n),k=n_per_age) 
                    for i in random_indexes:
                        cv2.imwrite(os.path.join(path_train,train_filenames[i]), train_imgs[i])
                        try:
                            label = label_dict[train_filenames[i]]
                            writer.writerow([train_filenames[i], label[0], label[1], label[2]])
                        except KeyError as e:
                            pass