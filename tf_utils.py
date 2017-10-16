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

# Thank you, @https://github.com/Pepslee
def max_unpool(net, corr_out, mask):
  assert mask is not None
  with tf.name_scope('UnPool2D'):
    input_shape = net.get_shape().as_list()
    output_shape = corr_out.get_shape().as_list()
    output_shape[-1] = input_shape[-1]
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int632), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret