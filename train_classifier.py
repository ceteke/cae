from datapy.data.datasets import CIFAR10Dataset, MNISTDataset, FashionDataset
from model import SWWAE
import tensorflow as tf
from utils import parse_layers
from arguments import get_emb_parser
import os
import numpy as np
from sklearn import svm, metrics

parser = get_emb_parser()
parsed = parser.parse_args()

if parsed.dataset == 'cifar10':
    dataset = CIFAR10Dataset()
    dataset.process()
    img_size = [32,32,3]
elif parsed.dataset == 'mnist':
    dataset = MNISTDataset()
    dataset.process()
    img_size = [28,28,1]
elif parsed.dataset == 'fashion':
    dataset = FashionDataset()
    dataset.process()
    img_size = [28, 28, 1]
else:
    print("Unknown dataset")
    exit()

dataset.sample_dataset()

if parsed.fc_layers is not None:
    fc_layers = [int(x) for x in parsed.fc_layers.split('-')]
else:
    fc_layers = []

layers = parse_layers(parsed.layer_str)

sess = tf.Session()
save_sess = tf.Session()
swwae = SWWAE(sess,img_size,'embedding',layers,fc_layers)

swwae.restore(os.path.join(parsed.out_dir))

X_train, y_train = dataset.get_batches(parsed.batch_size,shuffle=False)
X_test, y_test = dataset.get_batches(parsed.batch_size,train=False)
train_steps = len(X_train)
test_steps = len(X_test)

print("Forming training embedding matrix")

for train_step in range(train_steps):
    X_train_step = X_train[train_step]
    representation = swwae.get_representation(input=X_train_step)

    if train_step == 0:
        embedding_matrix = representation
        label_matrix = y_train[train_step]
    else:
        embedding_matrix = np.concatenate((embedding_matrix, representation))
        label_matrix = np.concatenate((label_matrix, y_train[train_step]))

print(embedding_matrix.shape)

print("Forming test embedding matrix")

for test_step in range(test_steps):
    X_test_step = X_test[test_step]
    representation = swwae.get_representation(input=X_test_step)

    if test_step == 0:
        test_embedding_matrix = representation
        test_label_matrix = y_test[test_step]
    else:
        test_embedding_matrix = np.concatenate((test_embedding_matrix, representation))
        test_label_matrix = np.concatenate((test_label_matrix, y_test[test_step]))

print(test_embedding_matrix.shape)

print("Training classifier")
clf = svm.SVC()
clf.fit(embedding_matrix, label_matrix)

print("Predicting")
y_pred = clf.predict(test_embedding_matrix)
acc = metrics.accuracy_score(test_label_matrix, y_pred)

print("Test acc:{}".format(acc))