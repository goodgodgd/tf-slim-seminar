# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
TF-Slim에서 variable을 사용하는 방법
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

"""
# nativeTF의 아래 코드와 TF-Slim의 아래 코드는 동일
with tf.device("/cpu:0"):
  weight_4 = tf.Variable(tf.truncated_normal(shape=[784, 200], mean=1.5, stddev=0.35), name="w4")
"""

weight_4 = slim.variable('w4',
                         shape=[784, 200],
                         initializer=tf.truncated_normal_initializer(mean=1.5, stddev=0.35),
                         device='/CPU:0')

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    val_w4 = sess.run(weight_4)
    print("w4 type:", type(val_w4))
    print("w4 values:", np.array_str(val_w4[:5, :10], precision=3, suppress_small=True))

"""
모델 변수(model variables) vs. 일반 변수(regular variable)
"""

# 모델 변수 생성하기
weight_5 = slim.model_variable('w5',
                               shape=[10, 10, 3, 3],
                               initializer=tf.truncated_normal_initializer(stddev=0.1),
                               regularizer=slim.l2_regularizer(0.05),
                               device='/CPU:0')

# native tf 처럼 모든 op 들의 이름을 받을 필요 없이 variable 들의 리스트만 받을 수 있다.
model_variables = slim.get_model_variables()
print("model variables: ", [var.name for var in model_variables])

# 일반 변수 생성하기
my_var_1 = slim.variable('mv1',
                         shape=[20, 1],
                         initializer=tf.zeros_initializer())

model_variables = slim.get_model_variables()
all_variables = slim.get_variables()

print("model variables: ", [var.name for var in model_variables])
print("all variables: ", [var.name for var in all_variables])


# tf-slim을 쓴다고 native tf를 쓰면 안되는 건 아니다. 섞어서 쓸수 있다.
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    # 대박;; 이름에 괄호를 치면 결과가 list로 나오고 그냥 넣으면 ndarray로 나옴
    val_w5_list = sess.run(['w5:0'])
    val_w5_arry = sess.run('w5:0')

    print('w5_list type: ', type(val_w5_list))
    print('w5_list value:', val_w5_list[0][0][0])
    print('w5_arry type: ', type(val_w5_arry))
    print('w5_arry value:', val_w5_arry[0,0,:,:])

