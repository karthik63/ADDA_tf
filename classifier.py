from __future__ import division
import os
from GAN import GAN
import time
import tensorflow as tf
import datetime
import numpy as np

from ops import *
from utils import *

encodings_train = np.loadtxt('train_gen_encodings.txt', delimiter=',')
labels_train = np.loadtxt('train_labels_complete.txt', delimiter=',')

encodings_test = np.loadtxt('test_gen_encodings.txt', delimiter=',')
labels_test = np.loadtxt('test_labels_complete.txt', delimiter=',')

images_test = np.load('test_images_mnist.npy')
image_labels_test = np.load('test_labels_mnist.npy')

images_train = np.load('train_images_mnist.npy')
image_labels_train = np.load('train_labels_mnist.npy')

images_valid = np.load('valid_images_mnist.npy')
image_labels_valid = np.load('valid_labels_mnist.npy')

is_training = True
load = False
batch_size = 128
n_epochs = 5
model='mnist'

encodings_batch = tf.placeholder(np.float32, [batch_size, 64])
labels_batch = tf.placeholder(np.float32, [batch_size, 5])

def build_classifier():

    global out, encodings_batch, labels_batch, c_vars_saver, loss, merged_sum

    with tf.variable_scope("classifier"):
        net = lrelu(tf.layers.dense(encodings_batch, 64, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    bias_initializer=tf.constant_initializer(0.0), trainable=is_training,
                                    name='hidden1'))

        net = lrelu(tf.layers.dense(net, 64, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    bias_initializer=tf.constant_initializer(0.0), trainable=is_training,
                                    name='hidden2'))

        out = tf.layers.dense(net, 5, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    bias_initializer=tf.constant_initializer(0.0), trainable=is_training,
                                    name='out')

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_batch, logits=out)

        c_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier/')

        print(c_vars)
        print("aay")

        c_vars_summary = []

        loss_summary = tf.summary.scalar('loss', tf.reduce_mean(loss))

        for var in c_vars:
            c_vars_summary.append(tf.summary.histogram(var.name, var))

        merged_sum = tf.summary.merge([loss_summary] + c_vars_summary)

        c_vars_saver = tf.train.Saver(var_list=c_vars, max_to_keep=5)


if not os.path.exists('logs_class'):
    os.mkdir('logs_class')

if not os.path.exists('checkpoints_class'):
    os.mkdir('checkpoints_class')


def train():
    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter('logs_class/' + datetime.datetime.now().isoformat(),
                                               sess.graph)

        build_classifier()

        optimise_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(loss)

        sess.run(tf.global_variables_initializer())

        batch_size = 128

        n_iterations = encodings_train.shape[0] // batch_size

        gan = GAN(sess, epoch=1, batch_size=batch_size, z_dim=64, dataset_name='mnist',
                  checkpoint_dir='error', result_dir='error',
                  log_dir='error', g_from_scratch='False')

        gan.build_model()

        gan.load_gen('checkpoints')

        best_so_far = 0
        global_step = 0

        for i in range(n_epochs):

            for j in range(n_iterations):

                # if global_step == 0:
                #     c_vars_saver.save(sess, 'checkpoints_class/classifier', 1)
                #     acc = mnist_test(sess, gan, build=False)
                #     if acc < best_so_far:
                #         best_so_far = acc

                global_step += 1

                _, merged_sum_value = sess.run([optimise_step, merged_sum],
                                         {encodings_batch: encodings_train[j*batch_size:(j+1)*batch_size],
                                          labels_batch: labels_train[j*batch_size:(j+1)*batch_size]})

                summary_writer.add_summary(merged_sum_value, global_step)

                if (j+1) % 100 == 0:
                    print(i, ' / ', n_epochs, '|||||', j, ' / ', n_iterations)

                    acc = mnist_test(sess, gan, restore=False, build=False)
                    if acc > best_so_far:
                        best_so_far = acc
                        c_vars_saver.save(sess, './checkpoints_class/classifier', global_step)





        print(best_so_far)



def svhn_test():

    with tf.Session() as sess:

        build_classifier()

        ckpt = tf.train.get_checkpoint_state('checkpoints_class')
        c_vars_saver.restore(sess, ckpt.model_checkpoint_path)

        n_iterations = encodings_test.shape[0] // batch_size

        prediction_count = 0
        correct_prediction_count = 0

        samples_per_class = np.zeros([5], np.int32)

        for j in range(n_iterations):

            prediction = sess.run(out, {encodings_batch: np.reshape(encodings_test[j:j+1], [batch_size, 64])})

            prediction = np.argmax(np.squeeze(prediction)).astype(np.int32)

            actual = np.argmax(np.squeeze(labels_test[j])).astype(np.int32)

            prediction_count += 1

            samples_per_class[actual] += 1

            if prediction == actual:
                correct_prediction_count += 1

    print(correct_prediction_count / prediction_count)
    print(samples_per_class)

def mnist_test(sess, gan, restore=True, build=True, batch_size=128, images_dataset=images_valid, labels_dataset=image_labels_valid):

    n_iterations = labels_dataset.shape[0] // batch_size

    prediction_count = 0
    correct_prediction_count = 0

    samples_per_class = np.zeros([5], np.int32)
    correct_per_class = np.zeros([5], np.int32)

    if build:
        build_classifier()

    if restore:
        ckpt = tf.train.get_checkpoint_state('checkpoints_class')
        c_vars_saver.restore(sess, ckpt.model_checkpoint_path)

    for j in range(n_iterations):

        ay = sess.run(gan.G, feed_dict={gan.inputs: images_dataset[j*batch_size: (j+1)*batch_size]})

        prediction = sess.run(out, {encodings_batch: ay})

        prediction = np.argmax(np.squeeze(prediction), axis=1).astype(np.int32)

        actual = np.squeeze(labels_dataset[j*batch_size: (j+1)*batch_size]).astype(np.int32)

        prediction_count += batch_size

        samples_per_class[actual] += 1

        # print(prediction[7], actual[7])

        whereall = np.where(prediction == actual)[0]
        correct_prediction_count += whereall.shape[0]
        correct_per_class[actual[whereall]] += 1

    accuracy = correct_prediction_count / prediction_count

    print(accuracy)
    # print(samples_per_class)
    # print(correct_per_class / samples_per_class)

    return accuracy


# train()

# with tf.Session() as sess:
#
#     gan = GAN(sess, epoch=1, batch_size=batch_size, z_dim=64, dataset_name='mnist',
#                       checkpoint_dir='error', result_dir='error',
#                       log_dir='error', g_from_scratch='False')
#
#     gan.build_model()
#
#     gan.load_gen('checkpoints')
#
#     acc = mnist_test(sess, gan, restore=True, build=True, images_dataset=images_test, labels_dataset=image_labels_test)

# svhn_test()