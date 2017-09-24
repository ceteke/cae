import tensorflow as tf

def _variable_on_cpu(name, shape, initializer, dtype):
    with tf.device('/cpu:0'):
        return tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=dtype)

def _variable_with_weight_decay(name, shape, stddev, wd, dtype):
    var = _variable_on_cpu(name=name, shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),
                           dtype=dtype)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var

def feed_encoder(input, dtype):
    # conv1
    with tf.variable_scope('conv1'):
        filter = _variable_with_weight_decay(name='weights', shape=[5, 5, 3, 64], stddev=5e-2, dtype=dtype, wd=0.0)
        conv = tf.nn.conv2d(input, filter=filter, strides=[1, 1, 1, 1], padding='SAME', name='conv2d')
        bias = _variable_on_cpu('bias', shape=[64], initializer=tf.constant_initializer(0.0), dtype=dtype)
        pre_activation = tf.nn.bias_add(conv, bias, name='pre_activation')
        conv1 = tf.nn.relu(pre_activation, 'activation')

    # pool1
    pool1_what, pool1_where = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv2
    with tf.variable_scope('conv2'):
        filter = _variable_with_weight_decay(name='weights', shape=[5, 5, 64, 64], stddev=5e-2, dtype=dtype, wd=0.0)
        conv = tf.nn.conv2d(conv1, filter=filter, strides=[1, 1, 1, 1], padding='SAME', name='conv2d')
        bias = _variable_on_cpu('bias', shape=[64], initializer=tf.constant_initializer(0.0), dtype=dtype)
        pre_activation = tf.nn.bias_add(conv, bias, name='pre_activation')
        conv2 = tf.nn.relu(pre_activation, 'activation')

    # pool2
    pool2_what, pool2_where = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    return pool1_what, pool1_where, pool2_what, pool2_where


