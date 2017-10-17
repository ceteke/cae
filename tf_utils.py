import tensorflow as tf

def variable_on_cpu(name, shape, initializer, dtype, trainable):
    with tf.device('/cpu:0'):
        return tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=dtype, trainable=trainable)

def variable_with_weight_decay(name, shape, stddev, wd, dtype, trainable):
    var = variable_on_cpu(name=name, shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),
                           dtype=dtype, trainable=trainable)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def max_pool_with_argmax(net, pool_size, stride):
  with tf.name_scope('MaxPoolArgMax'):
    _, mask = tf.nn.max_pool_with_argmax(
      net,
      ksize=[1, stride, stride, 1],
      strides=[1, stride, stride, 1],
      padding='VALID')
    mask = tf.stop_gradient(mask)
    net = tf.layers.max_pooling2d(net, pool_size, stride)
    return net, mask

def l2_regulazier(scale, collection_name):
    def l2(weights):
        tf.add_to_collection(collection_name, scale * tf.nn.l2_loss(weights, weights.op.name))
    return l2

def getwhere(y_prepool, y_postpool):
    ''' Calculate the 'where' mask that contains switches indicating which
    index contained the max value when MaxPool2D was applied.  Using the
    gradient of the sum is a nice trick to keep everything high level.'''
    return tf.gradients(tf.reduce_sum(y_postpool), y_prepool)[0]

# Thank you, @https://github.com/Pepslee
def max_unpool(net, corr, mask, pool_size):
  with tf.name_scope('UnPool2D'):
      corr_shape = corr.get_shape().as_list()
      out_shape = [corr_shape[1], corr_shape[2]]
      y = tf.image.resize_images(net, out_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      return tf.multiply(y, mask)
