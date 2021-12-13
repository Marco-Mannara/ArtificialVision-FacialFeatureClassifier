import os
import cv2

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

if __name__ == "__main__":
    path = "../dataset/utkface"
    dataset_filenames = os.listdir(path)
    age_counts = count_age_groups(dataset_filenames)

    #print(dataset_filenames)

    n_per_age = 250
    max_augmentation = 5
    
    dataset = []
    f = 0
    #cnt = 0
    for count in age_counts:
        age,n = count
        #group_filenames = dataset_filenames[cnt:cnt + n]
        #cnt += n
        group_filenames = [x for x in dataset_filenames if int(x.split("_")[0]) == age]
        group_imgs = []
        for filename in group_filenames:
            img = cv2.imread(os.path.join(path,filename))
            img = preprocessing(img)
            group_imgs.append(img)
        break
    cv2.waitKey(0)



    






