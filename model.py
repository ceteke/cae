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

def unravel_argmax(argmax, shape):
    output_list = [argmax // (shape[2]*shape[3]),
                   argmax % (shape[2]*shape[3]) // shape[3]]
    return tf.stack(output_list)

def max_unpool(bottom, argmax):
    bottom_shape = tf.shape(bottom)
    top_shape = [bottom_shape[0], bottom_shape[1]*2, bottom_shape[2]*2, bottom_shape[3]]

    batch_size = top_shape[0]
    height = top_shape[1]
    width = top_shape[2]
    channels = top_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat([t2, t3, t1], 4)
    indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

    x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))

def encoder_forward(input, dtype):
    # conv1
    with tf.variable_scope('conv1'):
        filter = _variable_with_weight_decay(name='weights', shape=[5, 5, 3, 64], stddev=5e-2, dtype=dtype, wd=0.0)
        conv = tf.nn.conv2d(input, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        bias = _variable_on_cpu('bias', shape=[64], initializer=tf.constant_initializer(0.0), dtype=dtype)
        pre_activation = tf.nn.bias_add(conv, bias, name='pre_activation')
        conv1 = tf.nn.relu(pre_activation, 'activation')

    # pool1
    pool1_what, pool1_where = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv2
    with tf.variable_scope('conv2'):
        filter = _variable_with_weight_decay(name='weights', shape=[5, 5, 64, 64], stddev=5e-2, dtype=dtype, wd=0.0)
        conv = tf.nn.conv2d(pool1_what, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        bias = _variable_on_cpu('bias', shape=[64], initializer=tf.constant_initializer(0.0), dtype=dtype)
        pre_activation = tf.nn.bias_add(conv, bias, name='pre_activation')
        conv2 = tf.nn.relu(pre_activation, 'activation')

    # pool2
    pool2_what, pool2_where = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    return pool1_what, pool1_where, pool2_what, pool2_where

def decoder_forward(pool1_what, pool1_where, pool2_what, pool2_where, lambda_M, dtype):
    #unpool1
    unpool1 = max_unpool(pool2_what, pool2_where)

    #decoder_conv1
    with tf.variable_scope('decoder_conv1'):
        filter = _variable_with_weight_decay(name='weights', shape=[5, 5, 64, 64], stddev=5e-2, dtype=dtype, wd=0.0)
        conv = tf.nn.conv2d(unpool1, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        bias = _variable_on_cpu('bias', shape=[64], initializer=tf.constant_initializer(0.0), dtype=dtype)
        pre_activation = tf.nn.bias_add(conv, bias, name='pre_activation')
        decoder_conv1 = tf.nn.relu(pre_activation)

    #L2M
    middle_loss = tf.multiply(lambda_M, tf.nn.l2_loss(tf.subtract(decoder_conv1, pool1_what)))
    tf.add_to_collection('losses', middle_loss)

    #unpool2
    unpool2 = max_unpool(decoder_conv1, pool1_where)

    #decoder_conv2
    with tf.variable_scope('decoder_conv2'):
        filter = _variable_with_weight_decay(name='weights', shape=[5, 5, 64, 3], stddev=5e-2, dtype=dtype, wd=0.0)
        conv = tf.nn.conv2d(unpool2, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        bias = _variable_on_cpu('bias', shape=[3], initializer=tf.constant_initializer(0.0), dtype=dtype)
        decoder_conv2 = tf.nn.bias_add(conv, bias)

    return decoder_conv2

def ae_loss(input, output, lambda_rec):
    reconstruction_loss = tf.multiply(lambda_rec, tf.nn.l2_loss(tf.subtract(input, output)))
    tf.add_to_collection('losses', reconstruction_loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')



