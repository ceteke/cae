from datapy.data.datasets import CIFAR10Dataset, MNISTDataset
from model import SWWAE
import tensorflow as tf
from utils import parse_layers
from arguments import get_emb_parser
import os
import numpy as np
from sklearn import svm, metrics

#parser = get_emb_parser()
#parsed = parser.parse_args()

dataset = MNISTDataset()
dataset.process()

print(dataset.test_data.shape)

clf = svm.LinearSVC()
clf.fit(dataset.training_data.reshape((55000,28*28)), dataset.training_labels)
y_pred = clf.predict(dataset.test_data.reshape((10000,28*28)))

acc = metrics.accuracy_score(dataset.test_labels, y_pred)

print("Test acc:{}".format(acc))