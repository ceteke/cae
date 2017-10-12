import tensorflow as tf

class SWWAE:
    def __init__(self, sess, image_shape, mode, layers, fc_ae_layers=None, fc_layers=None, learning_rate=None, lambda_rec=None,
                 lambda_M=None, dtype=tf.float32, tensorboard_id=None, num_classes=None, encoder_train=True, batch_size=32):
        self.layers = layers
        self.dtype = dtype
        self.mode = mode
        self.lambda_M = lambda_M
        self.lambda_rec = lambda_rec
        self.sess = sess
        self.learning_rate = learning_rate
        self.image_shape = image_shape
        self.tensorboard_id = tensorboard_id
        self.fc_layers = fc_layers
        self.num_classes = num_classes
        self.encoder_train = encoder_train
        self.fc_ae_layers = fc_ae_layers
        self.batch_size = batch_size

        self.form_variables()
        self.form_graph()

    def form_variables(self):
        self.input = tf.placeholder(shape=[self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]],
                                    dtype=self.dtype, name='input_batch')
        self.dropout_rate = tf.placeholder(shape=(), dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)

    def encoder_forward(self):
        encoder_whats = []
        encoder_what = self.input

        for i, layer in enumerate(self.layers):
            # convn
            with tf.variable_scope('conv{}'.format(i+1)):
                encoder_what = tf.layers.conv2d(encoder_what, layer.channel_size, layer.filter_size, padding='same')

            # pooln
            if layer.pool_size is not None:
                encoder_what = tf.layers.max_pooling2d(encoder_what, layer.pool_size, layer.pool_size,
                                                                      padding='same')

            encoder_whats.append(encoder_what)

        self.encoder_whats = encoder_whats
        pool_shape = encoder_whats[-1].get_shape()
        self.flatten = tf.reshape(encoder_whats[-1], [-1, (pool_shape[1] * pool_shape[2] * pool_shape[3]).value])

        if len(self.fc_ae_layers) == 0:
            self.representation = self.flatten
        else:
            self.encoder_fcs = []
            with tf.name_scope('encoder_fc'):
                for i, layer in enumerate(self.fc_ae_layers):
                    if i == 0:
                        encoder_fc = tf.layers.dense(self.flatten,self.fc_ae_layers[i], activation=tf.nn.relu)
                    else:
                        encoder_fc = tf.layers.dense(encoder_fc,self.fc_ae_layers[i], activation=tf.nn.relu)
                    self.encoder_fcs.append(encoder_fc)
            self.representation = encoder_fc

    def decoder_forward(self):
        if len(self.fc_ae_layers) == 0:
            decoder_what = self.encoder_whats[-1]
        else:
            with tf.name_scope('decoder_fc'):
                decoder_what = tf.layers.dropout(self.representation,self.dropout_rate)
                for i in range(len(self.fc_ae_layers)-1, -1, -1):
                    if i == 0:
                        decoder_what = tf.layers.dense(decoder_what,self.flatten.get_shape()[1].value,activation=tf.nn.relu)
                        fc_middle_loss = tf.multiply(self.lambda_M,
                                                     tf.nn.l2_loss(tf.subtract(decoder_what, self.flatten)))
                        tf.add_to_collection('losses', fc_middle_loss)
                    else:
                        decoder_what = tf.layers.dense(decoder_what,self.fc_ae_layers[i-1],activation=tf.nn.relu)

                        fc_middle_loss = tf.multiply(self.lambda_M,
                                                  tf.nn.l2_loss(tf.subtract(decoder_what, self.encoder_fcs[i - 1])))
                        tf.add_to_collection('losses', fc_middle_loss)

                pool_shape = self.encoder_whats[-1].get_shape()
                decoder_what = tf.reshape(decoder_what, [-1, pool_shape[1].value, pool_shape[2].value, pool_shape[3].value])

        decoder_whats = []
        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]

            with tf.variable_scope('deconv{}'.format(i+1)):
                if layer.pool_size is None:
                    if i == 0:  # Does not use non-linearity at the last layer
                        shape = self.image_shape[-1]
                        activation = None
                    else:
                        shape = self.layers[i - 1].channel_size
                        activation = tf.nn.relu

                    decoder_what = tf.layers.conv2d(decoder_what, shape, layer.filter_size, activation=activation,
                                                        padding='same',)
                else:
                    if i == 0: # Does not use non-linearity at the last layer
                        output_shape = self.input.get_shape()
                        filter_size = [layer.filter_size, layer.filter_size, self.image_shape[-1], layer.channel_size]
                        bias_size = self.image_shape[-1]
                    else:
                        up = self.encoder_whats[i-1]
                        output_shape = up.get_shape()
                        filter_size = [layer.filter_size, layer.filter_size, self.layers[i-1].channel_size, layer.channel_size]
                        bias_size = self.layers[i-1].channel_size

                    filter = tf.get_variable('filter', shape=filter_size, dtype=self.dtype,
                                             initializer=tf.glorot_uniform_initializer(dtype=self.dtype))
                    bias = tf.get_variable('bias', shape=bias_size, dtype=self.dtype,
                                           initializer=tf.constant_initializer(0.0, dtype=self.dtype))

                    decoder_what = tf.nn.conv2d_transpose(decoder_what, filter, output_shape,
                                                          strides=[1,layer.pool_size,layer.pool_size,1])
                    decoder_what = tf.nn.bias_add(decoder_what, bias)

                    if i != 0:
                        decoder_what = tf.nn.relu(decoder_what)

                decoder_whats.append(decoder_what)

            if i != 0:
                middle_loss = tf.multiply(self.lambda_M, tf.nn.l2_loss(tf.subtract(decoder_what, self.encoder_whats[i-1])))
                tf.add_to_collection('losses', middle_loss)

        self.decoder_what = decoder_what
        print(self.encoder_whats)
        print(decoder_whats)


    def ae_loss(self):
        reconstruction_loss = tf.multiply(self.lambda_rec, tf.nn.l2_loss(tf.subtract(self.input, self.decoder_what)))
        tf.add_to_collection('losses', reconstruction_loss)
        ae_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar('loss', ae_loss)
        return ae_loss

    def init_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt_op = optimizer.minimize(loss, global_step=self.global_step)

    def form_graph(self):
        print("Forming encoder", flush=True)
        self.encoder_forward()
        if self.mode == 'autoencode':
            print("Forming decoder", flush=True)
            self.decoder_forward()
            self.ae_loss = self.ae_loss()
            print("Forming L2 optimizer with learning rate {}".format(self.learning_rate), flush=True)
            self.init_optimizer(self.ae_loss)
            tf.summary.image('whatwhere/stacked', tf.concat((self.input, self.decoder_what), axis=2))

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('tensorboard/{}/train'.format(self.tensorboard_id), self.sess.graph)
        self.test_writer = tf.summary.FileWriter('tensorboard/{}/test'.format(self.tensorboard_id), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def train(self, input, labels=None):
        if self.mode == 'autoencode':
            _, batch_loss, global_step, tb_merge = self.sess.run([self.opt_op, self.ae_loss, self.global_step, self.merged],
                                                       feed_dict={self.input: input, self.dropout_rate: 0.5})
            self.train_writer.add_summary(tb_merge, global_step)
            return batch_loss, global_step

    def eval(self, input, labels=None):
        if self.mode == 'autoencode':
            loss, tb_merge, global_step = self.sess.run([self.ae_loss, self.merged, self.global_step], feed_dict={self.input:input, self.dropout_rate:0.0})
            self.test_writer.add_summary(tb_merge, global_step)
            return loss

    def get_representation(self, input):
        return self.sess.run(self.representation, feed_dict={self.input:input, self.dropout_rate:0.0})

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