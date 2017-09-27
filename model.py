import tensorflow as tf
from tf_utils import max_unpool, variable_on_cpu, variable_with_weight_decay

class SWWAE:
    def __init__(self, sess, image_shape, mode, layers, learning_rate, lambda_rec, lambda_M, dtype):
        self.layers = layers
        self.dtype = dtype
        self.mode = mode
        self.lambda_M = lambda_M
        self.lambda_rec = lambda_rec
        self.sess = sess
        self.learning_rate = learning_rate
        self.image_shape = image_shape

        self.form_variables()
        self.form_graph()

    def form_variables(self):
        self.input = tf.placeholder(shape=[None, self.image_shape[0], self.image_shape[1], self.image_shape[2]],
                                    dtype=self.dtype, name='input_batch')
        self.global_step = tf.Variable(0, trainable=False)
        if self.mode == 'classification':
            self.labels = tf.placeholder(shape=[32,], dtype=tf.int8, name='labels')

    def encoder_forward(self):
        encoder_whats = []
        encoder_wheres = []
        encoder_what = self.input

        for i, layer in enumerate(self.layers):
            # convn
            with tf.variable_scope('conv{}'.format(i+1)):
                if i == 0:
                    shape = [layer.filter_size, layer.filter_size, self.image_shape[-1], layer.channel_size]
                else:
                    shape = [layer.filter_size, layer.filter_size, self.layers[i-1].channel_size, layer.channel_size]

                filter = variable_with_weight_decay(name='weights',
                                                     shape=shape,
                                                     stddev=5e-2, dtype=self.dtype, wd=0.0)

                conv = tf.nn.conv2d(encoder_what, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
                bias = variable_on_cpu('bias', shape=[layer.channel_size], initializer=tf.constant_initializer(0.0),
                                        dtype=self.dtype)

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

        self.encoder_whats = encoder_whats
        self.encoder_wheres = encoder_wheres


    def decoder_forward(self):
        decoder_what = self.encoder_whats[-1]
        decoder_whats = []
        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            #unpooln
            if self.encoder_wheres[i] is not None:
                decoder_what = max_unpool(decoder_what, self.encoder_wheres[i],
                                          [1, layer.pool_size, layer.pool_size, 1],
                                          scope='unpool{}'.format(i+1))

            with tf.variable_scope('decoder_conv{}'.format(i+1)):
                if i == 0:
                    shape = [layer.filter_size, layer.filter_size, layer.channel_size, self.image_shape[-1]]
                    bias_size = self.image_shape[-1]
                else:
                    shape = [layer.filter_size, layer.filter_size, layer.channel_size, self.layers[i-1].channel_size]
                    bias_size = self.layers[i-1].channel_size

                filter = variable_with_weight_decay(name='weights',
                                                     shape=shape,
                                                     stddev=5e-2, dtype=self.dtype, wd=0.0)

                conv = tf.nn.conv2d(decoder_what, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
                bias = variable_on_cpu('bias', shape=[bias_size], initializer=tf.constant_initializer(0.0),
                                        dtype=self.dtype)

                pre_activation = tf.nn.bias_add(conv, bias, name='pre_activation')
                decoder_what = tf.nn.relu(pre_activation, 'activation')

                decoder_whats.append(decoder_what)

            if i != 0:
                middle_loss = tf.multiply(self.lambda_M, tf.nn.l2_loss(tf.subtract(decoder_what, self.encoder_whats[i-1])))
                tf.add_to_collection('losses', middle_loss)

        self.decoder_what = decoder_what

    def ae_loss(self):
        reconstruction_loss = tf.multiply(self.lambda_rec, tf.nn.l2_loss(tf.subtract(self.input, self.decoder_what)))
        tf.add_to_collection('losses', reconstruction_loss)
        self.ae_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    def init_optimizer(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.opt_op = optimizer.minimize(loss, global_step=self.global_step)

    def form_graph(self):
        print("Forming encoder", flush=True)
        self.encoder_forward()
        if self.mode == 'autoencode':
            print("Forming decoder", flush=True)
            self.decoder_forward()
            self.ae_loss()
            print("Forming optimizer with learning rate {}".format(self.learning_rate), flush=True)
            self.init_optimizer(self.ae_loss)
        self.sess.run(tf.global_variables_initializer())

    def train(self, input):
        if self.mode == 'autoencode':
            _, batch_loss, global_step = self.sess.run([self.opt_op, self.ae_loss, self.global_step],
                                                       feed_dict={self.input: input})
            return batch_loss, global_step

    def eval(self, input):
        if self.mode == 'autoencode':
            return self.sess.run(self.ae_loss, feed_dict={self.input:input})

    def save(self, path, ow=True):
        saver = tf.train.Saver()
        if ow:
            save_path = saver.save(self.sess, save_path=path)
        else:
            save_path = saver.save(self.sess, save_path=path, global_step=self.global_step)

        print('model saved at {}'.format(save_path), flush=True)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, save_path=path)