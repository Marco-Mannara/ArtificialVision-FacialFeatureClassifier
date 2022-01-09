import os
from tqdm import tqdm

from Dataset import Dataset
from FacialFeaturesClassifier import _preprocessing


dest_path = os.path.join("dataset", "preprocessed")

dset = Dataset("original", "dataset_manu")
dest_dset = Dataset("preprocessed", dest_path)

dset.load("Faces", "test_set.csv")
s_cnt = 0
cnt = 0

for name,img in tqdm(dset.samples.items(), desc = "Preprocessing images..."):
    img,succ = _preprocessing(img)
    if succ:
        s_cnt += 1
    dest_dset.samples[name] = img
    dest_dset.labels[name] = dset.labels[name]
    cnt +=1

try:
    os.mkdir(dest_path)
except OSError:
    pass

try:
    os.mkdir(os.path.join(dest_path,"samples"))
except OSError:
    pass


dest_dset.save("samples", "preproc_labels.csv")
print("Aligned: %.3f" % (round(float(s_cnt) / cnt, 3)))