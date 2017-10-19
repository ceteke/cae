import tensorflow as tf

def l2_regulazier(scale, collection_name):
    def l2(weights):
        tf.add_to_collection(collection_name, tf.multiply(tf.nn.l2_loss(weights), scale, name='L2_Reg'))
    return l2

def max_unpool(updates, mask, ksize=[1, 2, 2, 1], scope='unpool'):
    with tf.variable_scope(scope):
      input_shape = updates.get_shape().as_list()
      #  calculation new shape
      output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
      # calculation indices for batch, height, width and feature maps
      one_like_mask = tf.ones_like(mask)
      batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
      b = one_like_mask * batch_range
      y = mask // (output_shape[2] * output_shape[3])
      x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
      feature_range = tf.range(output_shape[3], dtype=tf.int64)
      f = one_like_mask * feature_range
      # transpose indices & reshape update values to one dimension
      updates_size = tf.size(updates)
      indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
      values = tf.reshape(updates, [updates_size])
      ret = tf.scatter_nd(indices, values, output_shape)
      return ret