import tensorflow as tf

def variable_on_cpu(name, shape, initializer, dtype):
    with tf.device('/cpu:0'):
        return tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=dtype)

def variable_with_weight_decay(name, shape, stddev, wd, dtype):
    var = variable_on_cpu(name=name, shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),
                           dtype=dtype)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var

def max_unpool(updates, mask, ksize, scope='unpool'):
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