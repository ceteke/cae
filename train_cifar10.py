from datapy.data.datasets import CIFAR10Dataset
from model import SWWAE
import tensorflow as tf
from utils import parse_layers

#dataset = CIFAR10Dataset()
#dataset.process()

#X, y = dataset.get_batches(32)

#print(X.shape, y.shape)

layers = parse_layers('(128)5c-2p-(128)3c-(256)3c-2p-(256)3c-2p')
sess = tf.Session()
swwae = SWWAE(sess,[32,32,3],'autoencode',layers,0.01,1.0,0.2,tf.float32)
#print(swwae.train(X[0]))