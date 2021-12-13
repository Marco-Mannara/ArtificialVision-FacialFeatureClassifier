import os, sys, csv, argparse
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def init_parameter():   
    parser = argparse.ArgumentParser(description='Final Project evaluation')
    parser.add_argument("--gt_path", type=str, default='gt.csv', help="File CSV con la groundtruth")
    parser.add_argument("--res_path", type=str, default='res.csv', help="File CSV con i risultati")
    args = parser.parse_args()
    return args


args = init_parameter()
# Lettura file CSV come dizionari (chiave path e beard, moustache e glasses valore)
# Groundtruth
with open(args.gt_path, mode='r') as csv_file:
    gt = csv.reader(csv_file, delimiter=',')
    gt_num = 0
    b_dict = {}
    m_dict = {}
    g_dict = {}
    for row in gt:
        #print(row)
        b_dict.update({row[0]: int(round(float(row[1])))})
        m_dict.update({row[0]: int(round(float(row[2])))})
        g_dict.update({row[0]: int(round(float(row[3])))})
        gt_num += 1
# Predizioni
with open(args.res_path, mode='r') as csv_file:
    res = csv.reader(csv_file, delimiter=',')
    res_num = 0
    b_res_dict = {}
    m_res_dict = {}
    g_res_dict = {}
    for row in res:
        #print(row)
        b_res_dict.update({row[0]: int(round(float(row[1])))})
        m_res_dict.update({row[0]: int(round(float(row[2])))})
        g_res_dict.update({row[0]: int(round(float(row[3])))})
        res_num += 1

print(gt_num)
print(res_num)

# Trasformazione in formato compatibile con scikit-learn
b_gt = []
m_gt = []
g_gt = []
b_res = []
m_res = []
g_res = []
for image in b_dict.keys():
    b_gt.append(b_dict[image])
    m_gt.append(m_dict[image])
    g_gt.append(g_dict[image])
    b_res.append(b_res_dict[image])
    m_res.append(m_res_dict[image])
    g_res.append(g_res_dict[image])

# Calcolo metriche di performance
beard_accuracy = accuracy_score(b_gt, b_res)
beard_balanced_accuracy = balanced_accuracy_score(b_gt, b_res)
moustache_accuracy = accuracy_score(m_gt, m_res)
moustache_balanced_accuracy = balanced_accuracy_score(m_gt, m_res)
glasses_accuracy = accuracy_score(g_gt, g_res)
glasses_balanced_accuracy = balanced_accuracy_score(g_gt, g_res)
avg_accuracy = (beard_accuracy + moustache_accuracy + glasses_accuracy) / 3
avg_balanced_accuracy = (beard_balanced_accuracy + moustache_balanced_accuracy + glasses_balanced_accuracy) / 3
fas = avg_accuracy + avg_balanced_accuracy
print("%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"%(beard_accuracy, beard_balanced_accuracy, moustache_accuracy, moustache_balanced_accuracy, 
    glasses_accuracy, glasses_balanced_accuracy, avg_accuracy, avg_balanced_accuracy, fas))