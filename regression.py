#!/usr/bin/env python3
# coding:utf-8
# ただのロジスティック回帰

# データのインポート
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os


# 損失関数をクロスエントロピーとする
def cross_entropy(y, y_):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    tf.summary.scalar('Entropy', cross_entropy)
    return cross_entropy


# 学習係数を指定して勾配降下アルゴリズムを用いてクロスエントロピーを最小化する
def training(cross_entropy):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    return train_step

# 変数をTensorBoardに出力する
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)  # 平均出力
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # Scalar出力(標準偏差)
        tf.summary.scalar('max', tf.reduce_max(var))  # Scalar出力(最大値)
        tf.summary.scalar('min', tf.reduce_min(var))  # Scalar出力(最小値)
        tf.summary.histogram('histogram', var)  # ヒストグラム出力

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

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# モデルの作成
# 画像データをxとする
with tf.name_scope('imageData'):
    x = tf.placeholder(tf.float32, [None, 784])

with tf.name_scope('Layer'):
    # モデルの重みをwと設定する
    with tf.name_scope('Weight'):
        W = tf.Variable(tf.zeros([784, 10]))
        variable_summaries(W)
    # モデルのバイアス
    with tf.name_scope('Bias'):
        b = tf.Variable(tf.zeros([10]))
        variable_summaries(b)
    # トレーニングデータ
    with tf.name_scope('Traindata'):
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        tf.summary.histogram('y', y)

# 正解のデータ
with tf.name_scope('Correct'):
    y_ = tf.placeholder(tf.float32, [None, 10])

# 変数の初期化
init = tf.global_variables_initializer()
# セッションの作成
with tf.Session() as sess:
    with tf.name_scope('CrossEntropy'):
        cross_entropy = cross_entropy(y, y_)

    # TensorBoardで追跡する変数を定義
    with tf.name_scope('Summary'):
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs', sess.graph)
        test_writer = tf.summary.FileWriter('./test')

    with tf.name_scope('Train'):
        training_op = training(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # 予測値と正解値を比較してbool値にする
            # argmax(y,1)は予測値の各行で最大となるインデックスをひとつ返す
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            # boole値を0もしくは1に変換して平均値をとる、これを正解率とする
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # セッションの開始及び初期化
    sess.run(init)

    # 学習
    for i in range(2000):
        # トレーニングデータからランダムに100個抽出する
        batch_xs, batch_ys = mnist.train.next_batch(100)

        # 10回に一回はテスト
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))

        # 100回に一回は詳細出力
        if i %100 == 99 :
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            # 確率的勾配降下法によりクロスエントロピーを最小化するような重みを更新する
            summary, _ = sess.run([merged, training_op],
                                  feed_dict={x: batch_xs, y_: batch_ys},
                                  options=run_options,
                                  run_metadata=run_metadata)
            writer.add_run_metadata(run_metadata, 'step%03d' % i)
            writer.add_summary(summary, _)
        else:
            # 確率的勾配降下法によりクロスエントロピーを最小化するような重みを更新する
            summary, _ = sess.run([merged, training_op], feed_dict={x: batch_xs, y_: batch_ys})
            writer.add_summary(summary, i)

    saver = tf.train.Saver()
    saver.save(sess, "./data/model_regression.ckpt")

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    writer.close()
    test_writer.close()
