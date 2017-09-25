from model import *
import tensorflow as tf
from utils import parse_layers

layers = parse_layers('(128)5c-2p-(128)3c-(256)3c-2p-(256)3c-2p')


input = tf.Variable(tf.truncated_normal([32, 200, 200, 3], dtype=tf.float32))
encoder_whats, encoder_wheres = encoder_forward(input, layers, tf.float32)
decoder_output = decoder_forward(encoder_whats, encoder_wheres, layers, lambda_M=0.2, dtype=tf.float32)
loss = ae_loss(input, decoder_output, lambda_rec=1.0)

# decoder_out = decoder_forward(pool1_what, pool1_where, pool2_what, pool2_where, lambda_M=0.2,dtype=tf.float32)
# loss = ae_loss(input, decoder_out, lambda_rec=1.0)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())