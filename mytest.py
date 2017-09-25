from model import *
import tensorflow as tf

input = tf.Variable(tf.truncated_normal([32, 200, 200, 3], dtype=tf.float32))
pool1_what, pool1_where, pool2_what, pool2_where = encoder_forward(input, tf.float32)
decoder_out = decoder_forward(pool1_what, pool1_where, pool2_what, pool2_where, lambda_M=0.2,dtype=tf.float32)
loss = ae_loss(input, decoder_out, lambda_rec=1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())