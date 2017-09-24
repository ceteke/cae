from model import encoder_forward
import tensorflow as tf

input = tf.Variable(tf.truncated_normal([32, 200, 200, 3], dtype=tf.float32))
pool1_what, pool1_where, pool2_what, pool2_where = encoder_forward(input, tf.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())