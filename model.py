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

def encoder_forward(input, layers, dtype):
    encoder_whats = []
    encoder_wheres = []
    encoder_what = input

    for i, layer in enumerate(layers):
        # convn
        with tf.variable_scope('conv{}'.format(i+1)):
            if i == 0:
                shape = [layer.filter_size, layer.filter_size, 3, layer.channel_size]
            else:
                shape = [layer.filter_size, layer.filter_size, layers[i-1].channel_size, layer.channel_size]

            filter = _variable_with_weight_decay(name='weights',
                                                 shape=shape,
                                                 stddev=5e-2, dtype=dtype, wd=0.0)

            conv = tf.nn.conv2d(encoder_what, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            bias = _variable_on_cpu('bias', shape=[layer.channel_size], initializer=tf.constant_initializer(0.0),
                                    dtype=dtype)

            pre_activation = tf.nn.bias_add(conv, bias, name='pre_activation')
            encoder_what = tf.nn.relu(pre_activation, 'activation')

        # pooln
        if layer.pool_size is not None:
            encoder_what, encoder_where = tf.nn.max_pool_with_argmax(encoder_what,
                                                                    ksize=[1, layer.pool_size, layer.pool_size, 1],
                                                                    strides=[1, 2, 2, 1], padding='SAME')
            encoder_wheres.append(encoder_where)

        else:
            encoder_wheres.append(None)

        encoder_whats.append(encoder_what)

    return encoder_whats, encoder_wheres


def decoder_forward(encoder_whats, encoder_wheres, layers, lambda_M, dtype):
    decoder_what = encoder_whats[-1]
    decoder_whats = []
    for i in range(len(layers)-1, -1, -1):
        layer = layers[i]
        #unpooln
        if encoder_wheres[i] is not None:
            decoder_what = max_unpool(decoder_what, encoder_wheres[i],
                                      [1, layer.pool_size, layer.pool_size, 1],
                                      scope='unpool{}'.format(i+1))

        with tf.variable_scope('decoder_conv{}'.format(i+1)):
            if i == 0:
                shape = [layer.filter_size, layer.filter_size, layer.channel_size, 3]
                bias_size = 3
            else:
                shape = [layer.filter_size, layer.filter_size, layer.channel_size, layers[i-1].channel_size]
                bias_size = layers[i-1].channel_size

            filter = _variable_with_weight_decay(name='weights',
                                                 shape=shape,
                                                 stddev=5e-2, dtype=dtype, wd=0.0)

            conv = tf.nn.conv2d(decoder_what, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            bias = _variable_on_cpu('bias', shape=[bias_size], initializer=tf.constant_initializer(0.0),
                                    dtype=dtype)

            pre_activation = tf.nn.bias_add(conv, bias, name='pre_activation')
            decoder_what = tf.nn.relu(pre_activation, 'activation')

            decoder_whats.append(decoder_what)

        if i != 0:
            middle_loss = tf.multiply(lambda_M, tf.nn.l2_loss(tf.subtract(decoder_what, encoder_whats[i-1])))
            tf.add_to_collection('losses', middle_loss)

    return decoder_what

def ae_loss(input, output, lambda_rec):
    reconstruction_loss = tf.multiply(lambda_rec, tf.nn.l2_loss(tf.subtract(input, output)))
    tf.add_to_collection('losses', reconstruction_loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')



