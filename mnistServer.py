#!/usr/bin/env python3
# coding:utf-8
# 手書き文字認識テストサーバー(楽なのでflaskで)

from flask import Flask,render_template,request,jsonify
import numpy as np
import tensorflow as tf
from mnist import model


x = tf.placeholder("float", [None, 784])
sess = tf.Session()

keep_prob = tf.placeholder("float")
y_conv, variables = model.convolution(x, keep_prob)
y = tf.nn.softmax(y_conv)
saver = tf.train.Saver(variables)
saver.restore(sess, "./data/model.ckpt")

ministServer = Flask(__name__)

def convolution(input):
    return sess.run(y, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

@ministServer.route('/')
def index():
    return render_template('index.html')

@ministServer.route('/sendCheckNum', methods=['POST'])
def sendCheckNum():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output = convolution(input)
    return jsonify(result=output)

if __name__ == '__main__':
    ministServer.run()

