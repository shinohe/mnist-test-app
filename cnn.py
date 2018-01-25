#!/usr/bin/env python3
# coding:utf-8

import os
import tensorflow as tf
from mnist import model

from tensorflow.examples.tutorials.mnist import input_data

# ディレクトリ内のファイルをすべて削除
def dir_clean(rootdir):
    for root, dirs, files in os.walk(rootdir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

# ログフォルダ内の削除
dir_clean('./logs')
dir_clean('./MNIST_data')

# データロード
data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# model
with tf.variable_scope("input"):
    # データ用可変2階テンソルを用意
    x = tf.placeholder("float", shape=[None, 784])
keep_prob = tf.placeholder(tf.float32)
y_conv, variables = model.convolution(x, keep_prob)

# train
with tf.name_scope('Correct'):
    y_ = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope('Entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # TensorBoardで追跡する変数を定義
    with tf.name_scope('Summary'):
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs', sess.graph)


    for i in range(2000):
        batch = data.train.next_batch(50)
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                      y_: batch[1],
                                                      keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        # 100回に一回は詳細出力
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, w_summary = sess.run([train_step, merged], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5},
                                    options=run_options,
                                    run_metadata=run_metadata)
            writer.add_summary(w_summary, _)

        else:
            _, w_summary = sess.run([train_step, merged], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            writer.add_summary(w_summary, i)

    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

    saver = tf.train.Saver(variables)
    path = saver.save(sess, "./data/model.ckpt", write_meta_graph=False, write_state=False)
    print("Saved:", path)

    writer.close()
