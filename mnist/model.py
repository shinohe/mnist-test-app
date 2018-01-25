#!/usr/bin/env python3
# coding:utf-8

import tensorflow as tf

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

# Multilayer Convolutional Network
def convolution(x, keep_prob):
    with tf.device("/cpu:0"):

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)


        # 画像をリシェイプ 第2引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
        with tf.variable_scope("imageData"):
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            tf.summary.image('preprocess', x_image, 10)

        # 一層目：畳み込み層
        with tf.name_scope('Layer1_conv'):
            # モデルの重みをwと設定する
            with tf.name_scope('Weight_conv1'):
                W_conv1 = weight_variable([5, 5, 1, 32])
                variable_summaries(W_conv1)
                tf.summary.image('filter', tf.transpose(W_conv1,perm=[3,0,1,2]), 10)
            # モデルのバイアス
            with tf.name_scope('Bias_conv1'):
                b_conv1 = bias_variable([32])
                variable_summaries(b_conv1)

            # 活性化関数うReLUで畳込み層を作成
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            
            # Tensorを[-1,28,28,32]から[-1,32,28,28]と順列変換し、[-1]と[-32]をマージしてimage出力
            tf.summary.image('convolved', tf.reshape(tf.transpose(h_conv1,perm=[0,3,1,2]),[-1,28,28,1]), 10)

        # 二層目：プーリング層
        with tf.name_scope('Layer2_pool'):
            h_pool1 = max_pool_2x2(h_conv1)
            
            # Tensorを[-1,14,14,32]から[-1,32,14,14]と順列変換し、[-1]と[32]をマージしてimage出力
            tf.summary.image('pooled', tf.reshape(tf.transpose(h_pool1,perm=[0,3,1,2]),[-1,14,14,1]), 10)

        # 三層目：畳み込み層
        with tf.name_scope('Layer3_conv'):
            with tf.name_scope('Weight_conv2'):
                W_conv2 = weight_variable([5, 5, 32, 64])
                variable_summaries(W_conv2)
                # Tensorを[5,5,32,64]から[32*64,5,5,1]と順列変換してimage出力
                tf.summary.image('filter', tf.reshape(tf.transpose(W_conv2,perm=[2,3,0,1]),[-1,5,5,1]), 20)

            with tf.name_scope('Bias_conv2'):
                b_conv2 = bias_variable([64])
                variable_summaries(b_conv2)

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

            # Tensorを[-1,14,14,64]から[-1,64,14,14]と順列変換し、[-1]と[64]をマージしてimage出力
            tf.summary.image('convolved', tf.reshape(tf.transpose(h_conv2,perm=[0,3,1,2]),[-1,14,14,1]), 10)



        # 四層目：畳み込み層
        with tf.name_scope('Layer4_pool'):
            h_pool2 = max_pool_2x2(h_conv2)
            # Tensorを[-1,7,7,64]から[-1,64,7,7]と順列変換し、[-1]と[64]をマージしてimage出力
            tf.summary.image('pooled', tf.reshape(tf.transpose(h_pool2,perm=[0,3,1,2]),[-1,7,7,1]), 10)

        # 五層目：フルコネクション層
        # わからない人への解説
        # 2x2ストライド2x2プーリングなので
        # 28 / 2 / 2 = 7
        # 出力の1024は割と適当
        with tf.name_scope('Layer5_fully_connected'):
            with tf.name_scope('Weight_fully_connected'):
                W_fc1 = weight_variable([7 * 7 * 64, 1024])
            with tf.name_scope('Bias_fully_connected'):
                b_fc1 = bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # ドロップアウト 過学習対策
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 六層目：出力層
        with tf.name_scope('Layer6_readout'):
            with tf.name_scope('Weight_readout'):
                W_rd = weight_variable([1024, 10])
                b_rd = bias_variable([10])

                y_conv = tf.matmul(h_fc1_drop, W_rd) + b_rd

    return y_conv, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_rd, b_rd]
