from datapy.data.datasets import CIFAR10Dataset, MNISTDataset
from model import SWWAE
import tensorflow as tf
from utils import parse_layers
from arguments import get_emb_parser
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import os

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

tf_path = parsed.save_path + '/embedding'

embedding_tensor = tf.stack(embedding_matrix, name='embedding')
embedding_tensor_variable = tf.Variable(embedding_tensor, trainable=False)
save_sess.run(tf.variables_initializer([embedding_tensor_variable]))
saver = tf.train.Saver(var_list=[embedding_tensor_variable])
saver.save(save_sess, save_path=tf_path)

meta_data = dataset.get_metadata()

with open(os.path.join(parsed.save_path, 'metadata.tsv'), 'w+') as f:
    f.writelines(meta_data)

sprite_path = os.path.join(parsed.save_path, 'sprite.png')
dataset.get_sprite(sprite_path)

config = projector.ProjectorConfig()

embedding = config.embeddings.add()
embedding.tensor_name = embedding_tensor_variable.name
embedding.metadata_path = os.path.join(parsed.save_path, 'metadata.tsv')

embedding.sprite.image_path = sprite_path
embedding.sprite.single_image_dim.extend([32, 32])

summary_writer = tf.summary.FileWriter(parsed.save_path)
projector.visualize_embeddings(summary_writer, config)