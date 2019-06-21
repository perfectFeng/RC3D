#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import os
import event_read
import c3d_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

save_dir = "chckPts/"
save_prefix = "save"
summaryFolderName = "summary/"

model_filename = "./chckPts/save27150.ckpt"
start_step = 0

batch_size = c3d_model.batch_size
num_frames = c3d_model.num_frames
height = c3d_model.height
width = c3d_model.width
channels = c3d_model.channels
n_classes = c3d_model.NUM_CLASSES

max_iters = 8


def _variable_with_weight_decay(name, shape, wd, trainable=True):
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var


def calc_reward(logit):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logit[:, :12])
    )
    iou_loss = tf.reduce_mean(tf.square(iou_placeholder - tf.sigmoid(logit[:, -1])))
    weight_decay_loss = tf.reduce_mean(tf.get_collection('weightdecay_losses'))

    total_loss = cross_entropy_mean + weight_decay_loss  + iou_loss
    tf.summary.scalar('total_loss', total_loss)
    return total_loss


def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit[:, :12], 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def evaluate():
    nextX, nextY = event_read.readTestFile(batch_size, num_frames)
    feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY}
    r = sess.run(accuracy, feed_dict=feed_dict)

    print("ACCURACY: " + str(r))


with tf.device('/gpu:1'):
    with tf.Graph().as_default():

        labels_placeholder = tf.placeholder(tf.int64, shape=batch_size)
        iou_placeholder = tf.placeholder(tf.float32, shape=batch_size)
        inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_frames, height, width, channels))

        with tf.variable_scope('var_name'):
            f_weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005, False),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005, False),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005, False),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005, False),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005, False),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005, False),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005, False),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005, False),
            }

            f_biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.000, False),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.000, False),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000, False),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000, False),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000, False),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000, False),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000, False),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000, False),
            }

        with tf.variable_scope('temp'):
            t_weights = {
                'twc1': _variable_with_weight_decay('twc1', [1, 3, 3, 512, 512], 0.0005, False),
                'twt1': _variable_with_weight_decay('twt1', [3, 1, 1, 512, 512], 0.0005, False),
                'tw2': _variable_with_weight_decay('tw2', [3, 1, 1, 512, 512], 0.0005),
                # 'tw3': _variable_with_weight_decay('tw3', [2, 1, 1, 512, 512], 0.0005),
            }
            t_biases = {
                'tbc1': _variable_with_weight_decay('tbc1', [512], 0.000, False),
                'tbt1': _variable_with_weight_decay('tbt1', [512], 0.000, False),
                'tb2': _variable_with_weight_decay('tb2', [512], 0.000),
                # 'tb3': _variable_with_weight_decay('tb3', [512], 0.000),
            }
        with tf.variable_scope('fc'):
            fc_weights = {
                'fcw1': _variable_with_weight_decay('fcw1', [8192, 4096], 0.0005, False),
                'fcw2': _variable_with_weight_decay('fcw2', [8192, 4096], 0.0005),
                # 'fcw3': _variable_with_weight_decay('fcw3', [8192, 4096], 0.0005),
            }
            fc_biases = {
                'fcb1': _variable_with_weight_decay('fcb1', [4096], 0.000, False),
                'fcb2': _variable_with_weight_decay('fcb2', [4096], 0.000),
                # 'fcb3': _variable_with_weight_decay('fcb3', [4096], 0.000),
            }
        with tf.variable_scope('out'):
            o_weights = {
                'ow1': _variable_with_weight_decay('ow1', [4096, n_classes], 0.0005, False),
                'ow2': _variable_with_weight_decay('ow2', [4096, n_classes], 0.0005),
                # 'ow3': _variable_with_weight_decay('ow3', [4096, n_classes], 0.0005),
            }
            o_biases = {
                'ob1': _variable_with_weight_decay('ob1', [n_classes], 0.000, False),
                'ob2': _variable_with_weight_decay('ob2', [n_classes], 0.000),
                # 'ob3': _variable_with_weight_decay('ob3', [n_classes], 0.000),
            }

        param = tf.trainable_variables()

        outputs = c3d_model.inference_c3d(inputs_placeholder, 0.5, f_weights, f_biases, t_weights, t_biases, fc_weights,
                                          fc_biases, o_weights, o_biases)

        loss = calc_reward(outputs)

        train_op = tf.train.GradientDescentOptimizer(1e-2).minimize(loss, var_list=param)
        # train_op = tf.group(train_op1, train_op2, train_op3)
        # null_op = tf.no_op()

        accuracy = tower_acc(outputs, labels_placeholder)
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)

            restore = tf.train.Saver({
                'var_name/wc1': f_weights['wc1'], 'var_name/bc1': f_biases['bc1'],
                'var_name/wc2': f_weights['wc2'], 'var_name/bc2': f_biases['bc2'],
                'var_name/wc3a': f_weights['wc3a'], 'var_name/bc3a': f_biases['bc3a'],
                'var_name/wc3b': f_weights['wc3b'], 'var_name/bc3b': f_biases['bc3b'],
                'var_name/wc4a': f_weights['wc4a'], 'var_name/bc4a': f_biases['bc4a'],
                'var_name/wc4b': f_weights['wc4b'], 'var_name/bc4b': f_biases['bc4b'],
                'var_name/wc5a': f_weights['wc5a'], 'var_name/bc5a': f_biases['bc5a'],
                'var_name/wc5b': f_weights['wc5b'], 'var_name/bc5b': f_biases['bc5b'],
                'temp/twc1': t_weights['twc1'], 'temp/tbc1': t_biases['tbc1'],
                'temp/twt1': t_weights['twt1'], 'temp/tbt1': t_biases['tbt1'],
                'fc/fcw1': fc_weights['fcw1'], 'fc/fcb1': fc_biases['fcb1'],
                'out/ow1': o_weights['ow1'], 'out/ob1': o_biases['ob1'],
            })
            saver.restore(sess, model_filename)

            summary_writer = tf.summary.FileWriter(summaryFolderName, graph=sess.graph)
            # training
            for epoch in range(max_iters):

                lines = event_read.readFile()

                for batch in range(int(len(lines) / batch_size)):

                    start_time = time.time()
                    nextX, nextY, ious = event_read.readTrainData(batch, lines, batch_size, num_frames)

                    feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY, iou_placeholder: ious}

                    _, summary, l, acc = sess.run([train_op, merged, loss, accuracy], feed_dict=feed_dict)

                    duration = time.time() - start_time

                    print('epoch-step %d-%d: %.3f sec' % (epoch, batch, duration))

                    if batch % 10 == 0:
                        saver.save(sess,
                                   save_dir + save_prefix + str(epoch * int(len(lines) / batch_size) + batch) + ".ckpt")
                        print('loss:', l, '---', 'acc:', acc)
                        summary_writer.add_summary(summary, epoch * int(len(lines) / batch_size) + batch)
                        evaluate()
