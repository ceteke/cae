from datapy.data.datasets import CIFAR10Dataset
from model import SWWAE
import tensorflow as tf
from utils import parse_layers
from arguments import get_emb_parser

parser = get_emb_parser()
parsed = parser.parse_args()

dataset = CIFAR10Dataset()
dataset.process()

layers = parse_layers(parsed.layer_str)

sess = tf.Session()
swwae = SWWAE(sess,[32,32,3],'embedding',layers)

