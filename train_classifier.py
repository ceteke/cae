from datapy.data.datasets import CIFAR10Dataset, MNISTDataset
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
else:
    dataset = MNISTDataset()
    dataset.process()
    img_size = [28,28,1]

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
train_steps = len(X_train)

print("Forming training embedding matrix")

for train_step in range(train_steps):
    X_train_step = X_train[train_step]
    representation = swwae.get_representation(input=X_train_step)

    if train_step == 0:
        embedding_matrix = representation
    else:
        embedding_matrix = np.concatenate((embedding_matrix, representation))

print(embedding_matrix.shape)

clf = svm.SVC(decision_function_shape='ovo', verbose=True)
clf.fit(embedding_matrix, dataset.training_labels)
y_pred = clf.predict(dataset.test_data)

acc = metrics.accuracy_score(dataset.test_labels, y_pred)

print("Test acc:{}".format(acc))