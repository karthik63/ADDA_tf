from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import sys
from collections import OrderedDict
from mnist import Mnist

imageSize = 28
channels = 1


def autoencoder(imageSize, channels, batchSize=None, isTrainable=True, reuse=False, other=0):
    """
    (?, 24, 24, 20)
    (?, 12, 12, 20)
    (?, 8, 8, 50)
    (?, 4, 4, 50)
    (?, 800)
    (?, 500)

    (?, 800)
    (?, 4, 4, 50)
    (?, 8, 8, 50)
    (?, 12, 12, 50)
    (?, 24, 24, 50)
    (?, 28, 28, 20)
    (?, 15680)
    (?, 784)

    (?, 28, 28, 64)
    (?, 14, 14, 64)
    (?, 14, 14, 64)
    (?, 7, 7, 64)
    (?, 7, 7, 64)
    (?, 3, 3, 64)
    (?, 3, 3, 64)
    (?, 1, 1, 64)
    (?, 64)
    (?, 64)

    """
    inputs = tf.placeholder(tf.float32, [batchSize, imageSize, imageSize, channels], name='inputs')

    layers = OrderedDict()
    print('isTrainable' + str(isTrainable))
    # encoder
    with tf.variable_scope('encoder') as scope:
        if reuse:
            scope.reuse_variables()
        rand_seed = 42
        net = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=3, padding='SAME', strides=1, \
                               kernel_initializer=tf.contrib.layers.xavier_initializer(seed=rand_seed), \
                               name='conv1', trainable=isTrainable, reuse=reuse)
        layers['conv1'] = net
        net = tf.layers.batch_normalization(net, center=True, scale=True, training=isTrainable, trainable=isTrainable,
                                            reuse=reuse, \
                                            name='batchnorm1')

        net = tf.nn.relu(net)
        print(net.shape)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='VALID', name='maxpool1')
        layers['pool1'] = net
        print(net.shape)

        net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=3, padding='SAME', strides=1, \
                               kernel_initializer=tf.contrib.layers.xavier_initializer(seed=rand_seed), \
                               name='conv2', trainable=isTrainable, reuse=reuse)
        layers['conv2'] = net
        net = tf.layers.batch_normalization(net, center=True, scale=True, training=isTrainable, trainable=isTrainable,
                                            reuse=reuse, \
                                            name='batchnorm2')
        net = tf.nn.relu(net)
        print(net.shape)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='VALID', name='maxpool2')
        layers['pool2'] = net
        print(net.shape)
        net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=3, padding='SAME', strides=1, \
                               kernel_initializer=tf.contrib.layers.xavier_initializer(seed=rand_seed), \
                               name='conv3', trainable=isTrainable, reuse=reuse)
        layers['conv3'] = net
        print(net.shape)
        net = tf.layers.batch_normalization(net, center=True, scale=True, training=isTrainable, trainable=isTrainable,
                                            reuse=reuse, \
                                            name='batchnorm3')
        net = tf.nn.relu(net)

        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='VALID', name='maxpool3')
        layers['pool3'] = net
        print(net.shape)
        net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=3, padding='SAME', strides=1, \
                               kernel_initializer=tf.contrib.layers.xavier_initializer(seed=rand_seed), \
                               name='conv4', trainable=isTrainable, reuse=reuse)
        layers['conv4'] = net
        print(net.shape)
        net = tf.layers.batch_normalization(net, center=True, scale=True, training=isTrainable, trainable=isTrainable,
                                            reuse=reuse, \
                                            name='batchnorm4')

        net = tf.nn.relu(net)

        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='VALID', name='maxpool4')
        layers['pool4'] = net
        print(net.shape)
        net = tf.contrib.layers.flatten(net)

        print(net.shape)
        net = tf.layers.dense(inputs=net, units=64, \
                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=rand_seed), \
                              activation=tf.nn.relu, name='fc1', trainable=isTrainable, reuse=reuse)
        layers['fc1'] = net
        print(net.shape)

        net = tf.reshape(net, [-1, 64])
        targetEnc = net

    # decoder
    with tf.variable_scope('decoder') as scope:
        if reuse:
            scope.reuse_variables()

        net = tf.layers.dense(inputs=net, units=64, activation=tf.nn.relu, name='decoder_fc', trainable=isTrainable, \
                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=rand_seed))
        print(net.shape)
        net = tf.reshape(net, [-1, 1, 1, 64])
        print(net.shape)
        # size = [int(net.shape[1] * 3), int(net.shape[2] * 3)]

        net = tf.image.resize_bilinear(net, size=[3, 3], align_corners=None, name='upsample1')
        print(net.shape)
        net = tf.contrib.layers.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=1, padding='SAME', \
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                                                     seed=rand_seed), \
                                                 activation_fn=None, trainable=isTrainable)
        net = tf.layers.batch_normalization(net, center=True, scale=True, training=isTrainable, trainable=isTrainable,
                                            reuse=reuse)

        net = tf.nn.relu(net)
        print(net.shape)
        # size = [int(net.shape[1] * 2), int(net.shape[2] * 2)]
        net = tf.image.resize_bilinear(net, size=[7, 7], align_corners=None, name='upsample2')
        print(net.shape)
        net = tf.contrib.layers.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=1, padding='SAME', \
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                                                     seed=rand_seed), \
                                                 activation_fn=None, trainable=isTrainable)
        net = tf.layers.batch_normalization(net, center=True, scale=True, training=isTrainable, trainable=isTrainable,
                                            reuse=reuse)

        net = tf.nn.relu(net)
        print(net.shape)
        # size = [int(net.shape[1] * 2), int(net.shape[2] * 2)]
        net = tf.image.resize_bilinear(net, size=[14, 14], align_corners=None, name='upsample3')
        print(net.shape)
        net = tf.contrib.layers.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=1, padding='SAME', \
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                                                     seed=rand_seed), \
                                                 activation_fn=None, trainable=isTrainable)
        net = tf.layers.batch_normalization(net, center=True, scale=True, training=isTrainable, trainable=isTrainable,
                                            reuse=reuse)

        net = tf.nn.relu(net)
        print(net.shape)
        # size = [int(net.shape[1] * 2), int(net.shape[2] * 2)]
        net = tf.image.resize_bilinear(net, size=[28, 28], align_corners=None, name='upsample4')
        print(net.shape)
        net = tf.contrib.layers.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=1, padding='SAME', \
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                                                     seed=rand_seed), \
                                                 activation_fn=None, trainable=isTrainable)
        net = tf.layers.batch_normalization(net, center=True, scale=True, training=isTrainable, trainable=isTrainable,
                                            reuse=reuse)

        net = tf.nn.relu(net)

        # net = tf.contrib.layers.flatten(net)
        print(net.shape)
        net = tf.contrib.layers.conv2d_transpose(net, num_outputs=1, kernel_size=1, stride=1, padding='SAME', \
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                                                     seed=rand_seed), \
                                                 activation_fn=tf.nn.relu, trainable=isTrainable)
        print(net.shape)

        # output = tf.layers.dense(inputs = net, units=28*28, trainable = isTrainable, activation=tf.nn.relu)
        output = tf.reshape(net, [-1, 28, 28, 1])
        print(output.shape)
        layers['output'] = output
    if other == 1:
        return inputs, output, targetEnc, layers
    loss = tf.losses.mean_squared_error(inputs, output)
    # if not isTrainable:
    #    return inputs,net,layers,loss

    print("Inside autoencoder")
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in params:
        print(var.name)
    # lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in params if 'bias' not in v.name ]) * 0.001
    # loss = loss+lossL2
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        # opt = tf.train.AdamOptimizer(0.001).minimize(loss)
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        gradsVars = opt.compute_gradients(loss, var_list=params)
        train = opt.apply_gradients(gradsVars)

    tf.summary.scalar("Loss", loss)
    for g, v in gradsVars:
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + str('grad'), g)

    tf.summary.image('Reconstructedimages', output, max_outputs=2)
    tf.summary.image('Originalimages', inputs, max_outputs=2)
    merged_all = tf.summary.merge_all()

    return inputs, output, layers, loss, train, merged_all



def train(data):
    # def train():
    max_iter_step = 50000
    batch_size = 256

    log_dir = 'logs_autoencoder_upd2/'
    model_dir = 'models_autoencoder_upd2/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        # autoencoder(imageSize,channels)
    inputs, net, layers, loss, opt, merged_all = autoencoder(imageSize, channels, batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Inside train")
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in params:
            print(var.name)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        saver = tf.train.Saver()
        prev_loss = 100000
        for i in range(max_iter_step):
            train_data = data.next_batch(batch_size)
            _, encoder_loss, merged = sess.run([opt, loss, merged_all], feed_dict={inputs: train_data})
            if (i % 10 == 0):
                summary_writer.add_summary(merged, i)
            print(prev_loss)
            print(encoder_loss)
            if abs(prev_loss - encoder_loss) < 0.00001:
                sys.exit(0)
            else:
                prev_loss = encoder_loss

            if (i % 1000 == 999):
                print("Iteration" + str(i) + ".......")
                print("Autoencoder Loss is:" + str(encoder_loss))

            if i % 1000 == 999:
                saver.save(sess, os.path.join(model_dir, 'models_' + str(i) + '.ckpt'))
                print("Model saved")


def restore_network(targetObj):
    reuse = False
    train = False
    batch_size = 256
    inputs, targetEnc, _ = autoencoder(imageSize, channels, batch_size, reuse=reuse, isTrainable=train, other=1)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        saver = tf.train.Saver(var_list=params)
        try:
            saver.restore(sess, 'models_autoencoder_upd2/' + 'models_' + str(49999) + '.ckpt')
        except:
            print("Previous weights not found")
            sys.exit(0)
        print("Model loaded")
        no_batches = 5

        targetTempEnc = np.array([])
        labelTempEnc = np.array([])
        for i in range(0, no_batches):
            targetData, targetLabels = targetObj.next_batch(batch_size, labelled=True)
            feedDict = {inputs: targetData}
            targetOut = sess.run([targetEnc], feed_dict=feedDict)
            targetTempEnc = np.concatenate((targetTempEnc, np.reshape(targetOut, (-1))), axis=0)
            targetLabel = np.argmax(targetLabels, axis=1) + 5

            labelTempEnc = np.concatenate((labelTempEnc, np.reshape(targetLabel, (-1))), axis=0)
        targetTempEnc = np.reshape(targetTempEnc, (-1, 64))
        labelTempEnc = np.reshape(labelTempEnc, (-1, 1))

        np.savetxt('targetEnc.txt', np.squeeze(np.array(targetTempEnc)), delimiter=',')
        np.savetxt('targetLabels.txt', np.array(labelTempEnc), delimiter=',')


def main():
    # data = Mnist(imageSize, channels)
    # train(data)
    restore_network(data)
    train()


if __name__ == "__main__":
    main()