import numpy as np
import pandas as pd
import random
import os
import cv2
import csv

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def get_tail_label(df):
    """
    Give tail label colums of the given target dataframe
    
    args
    df: pandas.DataFrame, target label df whose tail label has to identified
    
    return
    tail_label: list, a list containing column name of all the tail label
    """
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl)/irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label

def get_index(df):
  """
  give the index of all tail_label rows
  args
  df: pandas.DataFrame, target label df from which index for tail label has to identified
    
  return
  index: list, a list containing index number of all the tail label
  """
  tail_labels = get_tail_label(df)
  index = set()
  for tail_label in tail_labels:
    sub_index = set(df[df[tail_label]==1].index)
    index = index.union(sub_index)
  return list(index)

def get_minority_instace(X, y):
    """
    Give minority dataframe containing all the tail labels
    
    args
    X: pandas.DataFrame, the feature vector dataframe
    y: pandas.DataFrame, the target vector dataframe
    
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    index = get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub

def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs=NearestNeighbors(n_neighbors=5,metric='euclidean',algorithm='kd_tree').fit(X)
    euclidean,indices= nbs.kneighbors(X)
    return indices

def MLSMOTE(X,y, n_sample):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0,n-1)
        neighbour = random.choice(indices2[reference,1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val>2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbour,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target

if __name__=='__main__':
    
    path_dataset = os.path.join("dataset", "utkface")
    path_dataset_new = "dataset_smote"

    try:
        os.mkdir("dataset_smote")
    except OSError as err:
        pass

    filenames = os.listdir(path_dataset)
    label_dict = {}

    with open("labels.csv", "r") as label_file:
        reader = csv.reader(label_file)
        for row in tqdm(reader, desc="loading labels"):
            label_dict[row[0]] = [int(row[1]),int(row[2]),int(row[3])]

    imgs = np.zeros(shape = (len(filenames), 200 * 200 * 3), dtype='uint8')
    labels = np.zeros(shape = (len(filenames), 3), dtype='uint8')

    for i in tqdm(range(len(filenames)), desc = "loading imgs"):
        imgs[i][:] = cv2.imread(os.path.join(path_dataset,filenames[i])).flatten()
        try:
            labels[i][:] = label_dict[filenames[i]][:]
        except KeyError as err:
            pass
    
    #labels = pd.get_dummies(labels, prefix="pisellino")
    labels = pd.DataFrame(labels)
    imgs = pd.DataFrame(imgs)

    X_sub, y_sub = get_minority_instace(imgs, labels)   #Getting minority instance of that datframe
    print("Getting minority instance.")
    X_res,y_res =MLSMOTE(X_sub, y_sub, 100)     #Applying MLSMOTE to augment the dataframe
    print("Augmentation done.")

    X_res = X_res.to_numpy(dtype = 'uint8')
    X_res = np.reshape(X_res, (X_res.shape[0], 200, 200, 3))
    y_res = y_res.to_numpy()

    c = 0
    for x in tqdm(X_res, desc = "writing images to disk"):
        cv2.imwrite(os.path.join(path_dataset_new,str(c)+".jpg"),x)
        c+=1

    with open("new_labels.csv","w", newline = "") as file:
        writer = csv.writer(file)
        for y in y_res:
            writer.writerow(y)

    



