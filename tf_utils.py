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

# Thanks @ThomasWollmann for this
def max_unpool(pool, ind, ksize, scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape =  tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

        flat_input_size = tf.cumprod(input_shape)[-1]
        flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

        pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                          shape=tf.stack([input_shape[0], 1, 1, 1]))
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, tf.stack([flat_input_size, 1]))
        ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, tf.stack(output_shape))
        return ret