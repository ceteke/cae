import tensorflow as tf
from tf_utils import max_unpool, max_pool_with_argmax, l2_regulazier

class SWWAE:
    def __init__(self, sess, image_shape, mode, layers, rep_size=None, fc_layers=None, learning_rate=None, lambda_rec=None,
                 lambda_M=None, dtype=tf.float32, tensorboard_id=None, num_classes=None, encoder_train=True, batch_size=32,
                 sparsity=0.05, beta=0.5):
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
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.sparsity = sparsity
        self.beta = beta
        self.regulazier = l2_regulazier(0.01, collection_name='losses')
        self.kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1e-3)
        self.bias_initializer = tf.constant_initializer(0.0)
        self.form_variables()
        self.form_graph()

    def form_variables(self):
        self.input = tf.placeholder(shape=[self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]],
                                    dtype=self.dtype, name='input_batch')
        self.expected_output = tf.placeholder(shape=[self.batch_size] + self.image_shape, dtype=self.dtype, name='output_batch')
        self.train_time = tf.placeholder(shape=(), dtype=tf.bool)
        self.global_step = tf.Variable(0, trainable=False)

    def encoder_forward(self):
        encoder_wheres = []
        encoder_what = self.input

        for i, layer in enumerate(self.layers):
            # convn
            encoder_what = tf.layers.batch_normalization(encoder_what, training=self.train_time)
            with tf.variable_scope('conv{}'.format(i+1)):
                encoder_what = tf.layers.conv2d(encoder_what, layer.channel_size, layer.filter_size, padding='same',
                                                activation=tf.nn.relu, kernel_initializer=self.kernel_initializer,
                                                kernel_regularizer=self.regulazier, bias_initializer=self.bias_initializer)

            # pooln
            if layer.pool_size is not None:
                encoder_what, encoder_where = max_pool_with_argmax(encoder_what, layer.pool_size, layer.pool_size)
                encoder_wheres.append(encoder_where)

            else:
                encoder_wheres.append(None)

        pool_shape = encoder_what.get_shape()
        self.encoder_what = encoder_what
        self.flatten = tf.reshape(encoder_what, [-1, (pool_shape[1] * pool_shape[2] * pool_shape[3]).value])
        self.encoder_wheres = encoder_wheres

        if self.rep_size is None:
            self.representation = self.flatten
        else:
            with tf.name_scope('encoder_fc'):
                encoder_fc = tf.layers.dense(self.flatten,self.rep_size, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.regulazier, bias_initializer=self.bias_initializer)
                tf.summary.histogram('representation', encoder_fc)

                p_hat = tf.reduce_mean(encoder_fc, axis=0) # Mean over the batch
                p = tf.get_variable(name='p', shape=(self.rep_size), dtype=tf.float32, initializer=tf.constant_initializer(self.sparsity),
                                    trainable=False)
                one = tf.get_variable(name='1', shape=(self.rep_size), dtype=tf.float32, initializer=tf.constant_initializer(1.0),
                                      trainable=False)
                kl_divergence = tf.multiply(p, (tf.log(p) - tf.log(p_hat + 1e-3))) + tf.multiply(tf.subtract(one, p),
                                                                                          (tf.log(tf.subtract(one, p)) - tf.log(tf.subtract(one, p_hat) + 1e-3)))
                kl_divergence = tf.multiply(self.beta, tf.reduce_sum(kl_divergence), name='sparsity')
                # tf.add_to_collection('losses', kl_divergence)

            self.representation = encoder_fc

    def decoder_forward(self):
        if self.rep_size is None:
            decoder_what = self.encoder_what
        else:
            with tf.name_scope('decoder_fc'):
                decoder_what = tf.layers.dense(self.representation,self.flatten.get_shape()[1].value,kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.regulazier, bias_initializer=self.bias_initializer, activation=tf.nn.relu)

                pool_shape = self.encoder_what.get_shape()
                decoder_what = tf.reshape(decoder_what, [-1, pool_shape[1].value, pool_shape[2].value, pool_shape[3].value])

        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            #unpooln
            if self.encoder_wheres[i] is not None:
                decoder_what = max_unpool(decoder_what, self.encoder_wheres[i], layer.pool_size)

            with tf.variable_scope('deconv{}'.format(i+1)):
                if i == 0: # Does not use non-linearity at the last layer
                    shape = self.image_shape[-1]
                    decoder_what = tf.layers.conv2d_transpose(decoder_what, shape, layer.filter_size, padding='same',
                                                              kernel_initializer=self.kernel_initializer,
                                                              kernel_regularizer=self.regulazier,
                                                              bias_initializer=self.bias_initializer
                                                              )
                else:
                    shape = self.layers[i - 1].channel_size
                    decoder_what = tf.layers.conv2d_transpose(decoder_what, shape, layer.filter_size, padding='same',
                                                              activation=tf.nn.relu,
                                                              kernel_initializer=self.kernel_initializer,
                                                              kernel_regularizer=self.regulazier,
                                                              bias_initializer=self.bias_initializer
                                                              )

        self.decoder_what = decoder_what

    def ae_loss(self):
        reconstruction_loss = tf.nn.l2_loss(tf.subtract(self.expected_output, self.decoder_what),name='reconstruction')
        tf.add_to_collection('losses', reconstruction_loss)
        losses = tf.get_collection('losses')

        total_loss = tf.add_n(losses, name='total_loss')

        for l in losses + [total_loss]:
            loss_name = l.op.name
            tf.summary.scalar(loss_name, l)

        return total_loss

    def init_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt_op = optimizer.minimize(loss, global_step=self.global_step)

    def form_graph(self):
        print("Forming encoder", flush=True)
        self.encoder_forward()
        print("Forming decoder", flush=True)
        self.decoder_forward()
        self.loss = self.ae_loss()
        print("Forming L2 optimizer with learning rate {}".format(self.learning_rate), flush=True)
        self.init_optimizer(self.loss)
        tf.summary.image('whatwhere/stacked', tf.concat((self.input, self.decoder_what), axis=2))

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('tensorboard/{}/train'.format(self.tensorboard_id), self.sess.graph)
        self.test_writer = tf.summary.FileWriter('tensorboard/{}/test'.format(self.tensorboard_id), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def train(self, input, expected_output):
        _, batch_loss, global_step, tb_merge = self.sess.run([self.opt_op, self.loss, self.global_step, self.merged],
                                                       feed_dict={self.input: input, self.expected_output: expected_output, self.train_time:True})
        self.train_writer.add_summary(tb_merge, global_step)
        return batch_loss, global_step

    def eval(self, input, expected_output):
        loss, tb_merge, global_step = self.sess.run([self.loss, self.merged, self.global_step],
                                                        feed_dict={self.input:input, self.expected_output:expected_output, self.train_time:False})
        self.test_writer.add_summary(tb_merge, global_step)
        return loss

    def get_representation(self, input):
        return self.sess.run(self.representation, feed_dict={self.input:input, self.train_time:False})

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