'''
  MIT License

  Copyright (c) 2018 Kamran Janjua

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
'''

import os
import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import itertools, math
import pathlib
import tensorflow.contrib.layers as initializers
from scipy.spatial import distance

CENTER_LOSS_ALPHA = 0.5
NUM_CLASSES = 10
plt_range = 5

distArr = []
avgArr = []

threshold = 0.4
range_val = 2
slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('update_centers', 1000, 'numbers of steps after which update the centers.')
tf.app.flags.DEFINE_float('lambda_c', 1.0, 'The weight of the center loss')
tf.app.flags.DEFINE_float('lambda_g', 1.0, 'The weight of the git loss')
tf.app.flags.DEFINE_integer('gpu', 0, 'GPU index to be used')
FLAGS = tf.app.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

epoch = 0
counter = 0


def get_centers(feat_list, label_list):
    centers_list = []
    for idx in range(10):
        list_of_indices = [n for n, x in enumerate(label_list) if x == idx]

        items_of_class = []
        for item in list_of_indices:
            got_feat = [float(i) for i in feat_list[item]]
            items_of_class.append(got_feat)

        mean = np.mean(items_of_class, axis=0)
        centers_list.append(mean)
    return np.asarray(centers_list)


def get_intra_class_distance(feat_lst, label_lst, centers):
    distances_list = []
    for idx in range(10):
        list_of_indices = [n for n, x in enumerate(label_lst) if x == idx]

        list_for_class = []
        for item in list_of_indices:
            got_feat = [float(i) for i in feat_lst[item]]
            list_for_class.append(got_feat)

        distance_feat_from_center = []
        for item in list_for_class:
            distance_feat_from_center.append(distance.euclidean(item, centers[idx]))
        intraclass_distance = np.mean(distance_feat_from_center, axis=0)
        distances_list.append(intraclass_distance)
    return distances_list


with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='input_images')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')

global_step = tf.Variable(0, trainable=False, name='global_step')

def get_distances(features, labels, num_classes):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)

    # distances = features - centers_batch
    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = tf.divide(diff, tf.cast((1 + appear_times), tf.float32))

    return diff


def get_git_loss(features, labels, num_classes):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)

    loss = tf.reduce_mean(tf.square(features - centers_batch))

    # Pairwise differences
    diffs = (features[:, tf.newaxis] - centers_batch[tf.newaxis, :])
    diffs_shape = tf.shape(diffs)

    # Mask diagonal (where i == j)
    mask = 1 - tf.eye(diffs_shape[0], diffs_shape[1], dtype=diffs.dtype)
    diffs = diffs * mask[:, :, tf.newaxis]

    # combinaton of two losses
    loss2 = tf.reduce_mean(tf.divide(1, 1 + tf.square(diffs)))

    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = tf.divide(diff, tf.cast((1 + appear_times), tf.float32))
    diff = CENTER_LOSS_ALPHA * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)  # diff is used to get updated centers.

    # combo_loss = value_factor * loss + new_factor * loss2
    combo_loss = FLAGS.lambda_c * loss + FLAGS.lambda_g * loss2

    return combo_loss, centers_update_op


def inference(input_images):
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):
            x = slim.conv2d(input_images, num_outputs=32, weights_initializer=initializers.xavier_initializer(),
                            scope='conv1_1')
            x = slim.conv2d(x, num_outputs=32, weights_initializer=initializers.xavier_initializer(), scope='conv1_2')
            x = slim.max_pool2d(x, scope='pool1')
            x = slim.conv2d(x, num_outputs=64, weights_initializer=initializers.xavier_initializer(), scope='conv2_1')
            x = slim.conv2d(x, num_outputs=64, weights_initializer=initializers.xavier_initializer(), scope='conv2_2')
            x = slim.max_pool2d(x, scope='pool2')
            x = slim.conv2d(x, num_outputs=128, weights_initializer=initializers.xavier_initializer(), scope='conv3_1')
            x = slim.conv2d(x, num_outputs=128, weights_initializer=initializers.xavier_initializer(), scope='conv3_2')
            x = slim.max_pool2d(x, scope='pool3')
            x = slim.flatten(x, scope='flatten')
            feature = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc1')
            x = tflearn.prelu(feature)
            x = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc2')
    return x, feature


def build_network(input_images, labels):
    logits, features = inference(input_images)

    with tf.variable_scope('loss') as scope:
        # with tf.name_scope('center_loss'):
        #     center_loss, centers_update_op_int = get_center_loss(features, labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        with tf.name_scope('git_loss'):
            git_loss, centers_update_op_int = get_git_loss(features, labels, NUM_CLASSES)
        scope.reuse_variables()
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss + git_loss

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))

    with tf.name_scope('loss/'):
        # tf.summary.scalar('CenterLoss', center_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)

    with tf.name_scope('dist'):
        distances_op = get_distances(features, labels, NUM_CLASSES)

    return logits, features, total_loss, accuracy, centers_update_op_int, distances_op  # returns total loss


logits, features, total_loss, accuracy, centers_update_op, distances_op = build_network(input_images, labels)
mnist = input_data.read_data_sets('/tmp/mnist', reshape=False)
optimizer = tf.train.AdamOptimizer(0.001) # learning rate. 
train_op = optimizer.minimize(total_loss, global_step=global_step)

summary_op = tf.summary.merge_all()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
writer = tf.summary.FileWriter('maxi-fig', sess.graph)
mean_data = np.mean(mnist.train.images, axis=0)
step = sess.run(global_step) + 1

exp_save_dir = 'reduce_mean-updatec_%d_lc_%f_lg_%f' % (FLAGS.update_centers, FLAGS.lambda_c, FLAGS.lambda_g)
pathlib.Path(exp_save_dir).mkdir(parents=True, exist_ok=True)
batch_size = 128
intra_cls_dist = 0
vali_acc = 0
inter_cls_dist = 0
with open(exp_save_dir + "/results.txt", "w") as text_file:
    while step < 8000:
        batch_images, batch_labels = mnist.train.next_batch(batch_size)
        _, summary_str, train_acc, train_loss, updated_centers = sess.run(
            [train_op, summary_op, accuracy, total_loss, centers_update_op],
            feed_dict={
                input_images: batch_images - mean_data,
                labels: batch_labels,
            })

        step += 1
        writer.add_summary(summary_str, global_step=step)
        if step % FLAGS.update_centers == 0:

            num_train_samples = mnist.train.num_examples
            num_of_batches = num_train_samples // batch_size
            centers = np.zeros([NUM_CLASSES, 2])
            all_features = []
            all_labels = []
            for b in range(num_of_batches):
                batch_images, batch_labels = mnist.train.next_batch(batch_size, shuffle=False)
                feat2 = sess.run(features, feed_dict={input_images: batch_images - mean_data})
                all_features.extend(feat2)
                all_labels.extend(batch_labels)
                c = get_centers(feat2, batch_labels)
                centers = np.sum(np.array([centers, c]), axis=0)

            centers = centers / num_of_batches

            d = get_intra_class_distance(all_features, all_labels, centers)
            # print(d)
            intra_cls_dist = np.mean(np.asarray(d))
            print("intra class distance %f" % intra_cls_dist)

            for i, j in itertools.combinations(centers, 2):
                distance1 = math.sqrt(((i[0] - j[0]) ** 2) + ((i[1] - j[1]) ** 2))
                distArr.append(distance1)
            inter_cls_dist = float(sum(distArr)) / len(distArr)
            avgArr.append(inter_cls_dist)
            print("The average distance between two centers is: ", inter_cls_dist)

            print(("Step: {}, Loss: {:.4f}".format(step, train_loss)))  # prints training loss and steps.
            epoch += 1
            vali_image = mnist.validation.images - mean_data
            vali_acc = sess.run(
                accuracy,
                feed_dict={
                    input_images: vali_image,
                    labels: mnist.validation.labels
                })

            print("\nEpochs Done: {}".format(epoch))
            print("{} Steps Done.".format(step))
            print(("Step: {}, Train_Acc:{:.4f}, Valid_Acc:{:.4f}".
                   format(step, train_acc, vali_acc)))
            print("\n")
            print("====================================================")
            # li = (sess.run(centers_update_op, feed_dict={
            #     input_images: batch_images - mean_data,
            #     labels: batch_labels,
            # }))
            # feat_samp = (sess.run(features, feed_dict={
            #     input_images: batch_images - mean_data,
            #     labels: batch_labels,
            # }))
            # print("Features", feat_samp)

            text_file.write(
                (
                    "Step:\t{}, Loss:\t{:.4f}, Train_Acc:\t{:.4f}, Valid_Acc:\t{:.4f}, inter_cls_dist:\t{:.4f}, intra_cls_dist:\t{:.4f}\n".
                    format(step, train_loss, train_acc, vali_acc, inter_cls_dist, intra_cls_dist)))
            n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            save_path = saver.save(sess, "model/model-maxi/model-epoch{}.ckpt".format(
                epoch))
            print("Model saved for epoch # {}".format(epoch))

            # Graphing the end results.
            feat2 = sess.run(features, feed_dict={input_images: mnist.train.images[:1000] - mean_data})
            labels2 = mnist.train.labels[:1000]
            # centers = get_centers(feat2, labels2)

            f = plt.figure(num=2)
            # ax = plt.gca()
            # ax.set_xlim([-10, 11])
            # ax.set_ylim([-13, 5])
            c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
                 '#ff00ff', '#990000', '#999900', '#009900', '#009999']
            for i in range(10):
                plt.plot(feat2[labels2 == i, 0].flatten(), feat2[labels2 == i, 1].flatten(), '.', c=c[i])
            plt.scatter(*zip(*centers), color='black')  # To plot the centers on the datasets.
            # plt.plot(*zip(*li), color='red')  # connect all the clusters by a line.
            plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            plt.grid()
            plt.savefig(exp_save_dir + '/Cluster-Results-{}.png'.format(epoch))
            # plt.show()
            plt.close(2)

        if step == 8000 - 1:
            with open("results.txt", "a") as text_file:
                text_file.write(
                    ("lc:\t{}, lg:\t{}, Step:\t{}, Loss:\t{:.4f}, Train_Acc:\t{:.4f}, Valid_Acc:\t{:.4f}, "
                     "inter_cls_dist:\t{:.4f}, intra_cls_dist:\t{:.4f}\n".
                     format(FLAGS.lambda_c, FLAGS.lambda_g, step, train_loss, train_acc, vali_acc, inter_cls_dist,
                            intra_cls_dist)))
