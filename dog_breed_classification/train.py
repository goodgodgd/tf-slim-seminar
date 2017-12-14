# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from utils.dataset_utils import load_batch
from datasets import tf_record_dataset

tf.logging.set_verbosity(tf.logging.INFO)


'''
# 훈련 데이터 로드
'''
batch_size = 16
tfrecord_dataset = tf_record_dataset.TFRecordDataset(
    tfrecord_dir='/home/itrocks/Git/Tensorflow/dog-breed-classification.tf/raw_data/dog/tfrecord',
    dataset_name='dog',
    num_classes=120)
# Selects the 'train' dataset.
dataset = tfrecord_dataset.get_split(split_name='train')
images, labels, num_samples = load_batch(dataset, batch_size=batch_size, height=224, width=224)


'''
# 네트워크 모델 로드: VGG-16
'''
vgg = tf.contrib.slim.nets.vgg
with slim.arg_scope(vgg.vgg_arg_scope()):
  logits, end_points = vgg.vgg_16(inputs=images, num_classes=120, is_training=True)


'''
# 체크포인트로부터 파라미터 복원하기
'''
# 다운로드한 VGG-16 체크포인트 파일 경로
model_path = '/home/itrocks/Backup/Model/TF-Slim/vgg_16.ckpt'

# 마지막 fc8 레이어는 파라미터 복원에서 제외
exculde = ['vgg_16/fc8']
variables_to_restore = slim.get_variables_to_restore(exclude=exculde)
# saver = tf.train.Saver(variables_to_restore)
# with tf.Session() as sess:
#   saver.restore(sess, model_path)

init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore, ignore_missing_vars=True)


'''
# 손실함수 정의
'''
loss = slim.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
total_loss = slim.losses.get_total_loss()


'''
# 옵티마이저 정의: Adam
# learrning rate decay 적용
'''
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.9
num_epochs_before_decay = 2

global_step = get_or_create_global_step()

num_batches_per_epoch = num_samples / batch_size
num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)
lr = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                global_step=global_step,
                                decay_steps=decay_steps,
                                decay_rate=learning_rate_decay_factor,
                                staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=lr)


'''
# 메트릭 정의
'''
predictions = tf.argmax(logits, 1)
targets = tf.argmax(labels, 1)

correct_prediction = tf.equal(predictions, targets)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('losses/Total', total_loss)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()


'''
# 훈련하기
'''
# logging 경로 설정
logdir = '/home/itrocks/Downloads/dog_model/'

if not tf.gfile.Exists(logdir):
  tf.gfile.MakeDirs(logdir)

# 훈련 오퍼레이션 정의
train_op = slim.learning.create_train_op(total_loss, optimizer)

final_loss = slim.learning.train(train_op=train_op,
                                 logdir=logdir,
                                 init_fn=init_fn,
                                 number_of_steps=500000,
                                 summary_op=summary_op,
                                 save_summaries_secs=300,
                                 save_interval_secs=600)

print('Finished training. Final batch loss %f' %final_loss)