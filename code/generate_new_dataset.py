from hashlib import new
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler

# LOAD THE DHG DATASET
objects = []

with (open("dhg_data.pckl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break


# LOAD THE NEW GESTURE
temp = []
gesture = []

for i in range(1,31):
    temp = np.loadtxt('gesture-capture-{}.csv'.format(i), delimiter=',', dtype=float)
    gesture.extend([temp])


# CREATE NEW DATASET
new_dataset = []
labels = []
count = np.zeros(15, dtype=int)


# APPENDING THE DHG DATASET
for i in range(len(objects[0]['y_train_14'])):

    label = objects[0]['y_train_14'][i] - 1
    num = count[label]

    if num < 90:

        count[label] = count[label] + 1
        new_dataset.extend([objects[0]['x_train'][i]])
        labels.append(label+1)

for i in range(len(objects[0]['y_test_14'])):

    label = objects[0]['y_test_14'][i] - 1

    if label > 13:
        continue

    num = count[label]

    if num < 100:

        count[label] = count[label] + 1
        new_dataset.extend([objects[0]['x_test'][i]])
        labels.append(label+1)


# APPENDING THE NEW GESTURE
new_dataset.extend(gesture)

for i in range(30):
    labels.append(15)


# TRANSFORMING IT INTO TRAIN SET (90%) AND TEST SET (10%)
full_dataset = []
for i in range(len(new_dataset)):
   full_dataset.append([new_dataset[i], labels[i]])

train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [(train_size), test_size])

x_train = []
y_train = []
x_test = []
y_test = []

for batch_index, (faces, labels) in enumerate(train_dataset):
    x_train.append(faces)
    y_train.append(labels)

for batch_index, (faces, labels) in enumerate(test_dataset):
    x_test.append(faces)
    y_test.append(labels)


# CREATE DICT FOR THE PICKLE FILE
data = {'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test}


# CREATE A PICKLE FILE
with open('new_dataset.pkl', 'wb') as handle:
    pickle.dump(data, handle)

# OPEN THE PICKLE FILE
# with open('new_dataset.pkl', 'rb') as handle:
#     read_back = pickle.load(handle)