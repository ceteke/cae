from datapy.data.datasets import CIFAR10Dataset
from model import SWWAE
import tensorflow as tf
from utils import parse_layers
from arguments import get_emb_parser
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import os

parser = get_emb_parser()
parsed = parser.parse_args()

dataset = CIFAR10Dataset()
dataset.process()

layers = parse_layers(parsed.layer_str)

sess = tf.Session()
save_sess = tf.Session()
swwae = SWWAE(sess,[32,32,3],'embedding',layers)

swwae.restore('output/swwae')

X_test, _ = dataset.get_batches(parsed.batch_size, train=False)
test_steps = len(X_test)

print("Forming embedding matrix")

for test_step in range(test_steps):
    X_test_step = X_test[test_step]
    representation = swwae.get_representation(input=X_test_step)

    if test_step == 0:
        embedding_matrix = representation
    else:
        embedding_matrix = np.concatenate((embedding_matrix, representation))

print(embedding_matrix.shape)

embedding_tensor = tf.stack(embedding_matrix, name='embedding')
embedding_tensor_variable = tf.Variable(embedding_tensor, trainable=False)
save_sess.run(tf.variables_initializer([embedding_tensor_variable]))
saver = tf.train.Saver(var_list=[embedding_tensor_variable])
saver.save(save_sess, save_path=parsed.save_path)

meta_data = dataset.get_metadata()

with open(os.path.join(parsed.save_path, 'metadata.tsv'), 'w+') as f:
    f.writelines(meta_data)

summary_writer = tf.train.SummaryWriter(parsed.save_path)
config = projector.ProjectorConfig()

embedding = config.embeddings.add()
embedding.tensor_name = embedding_tensor_variable.name
embedding.metadata_path = os.path.join(parsed.save_path, 'metadata.tsv')
projector.visualize_embeddings(summary_writer, config)