# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
모델 평가하기
MNIST examples
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tf_slim_tutorial.c01_defining_models.s04_examples.mnist_deep_step_by_step_slim as mnist_model
from tf_slim_tutorial.utils.dataset_utils import load_batch
from tf_slim_tutorial.datasets import tf_record_dataset

tf.logging.set_verbosity(tf.logging.INFO)

'''
# 평가 데이터 로드
'''
mnist_tfrecord_dataset = tf_record_dataset.TFRecordDataset(
    tfrecord_dir='C:/Work/tfslim-tutorial/data/mnistasjpg/trainingSet/tfrecord',
    dataset_name='mnist',
    num_classes=10)
# Selects the 'train' dataset.
dataset = mnist_tfrecord_dataset.get_split(split_name='validation')
images, labels, _ = load_batch(dataset)

'''
# 모델 정의
'''
predictions = mnist_model.mnist_convnet(inputs=images, is_training=False)

'''
# 메트릭 정의
'''
predictions = tf.argmax(predictions, 1)
labels = tf.argmax(labels, 1)

# Define the metrics:
# slim.metrics.streaming_accuracy() returns 'accuracy' and 'update_op'
# slim.metrics.aggregate_metric_map() just pairs name with two ops
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
    # 'eval/Recall@5': slim.metrics.streaming_recall_at_k(predictions, labels, 5),
})

'''
# 평가하기
'''
# logging 경로 설정
log_dir = 'C:/Work/tfslim-tutorial/model/vggmnist'
eval_dir = 'C:/Work/tfslim-tutorial/model/vggmnist-eval'
if not tf.gfile.Exists(eval_dir):
    tf.gfile.MakeDirs(eval_dir)

if not tf.gfile.Exists(log_dir):
    raise Exception("trained check point does not exist at %s " % log_dir)
else:
    checkpoint_path = tf.train.latest_checkpoint(log_dir)

'''
eval_op: A operation run `num_evals` times.
final_op: An operation to execute after all of the `eval_op` executions. The value of `final_op` is returned.
'''

metric_values = slim.evaluation.evaluate_once(
    master='',
    checkpoint_path=checkpoint_path,
    logdir=eval_dir,
    num_evals=100,
    eval_op=list(names_to_updates.values())[0],
    final_op=list(names_to_values.values())[0])

print(list(names_to_values.keys())[0], metric_values)
# names_to_values = dict(zip(names_to_values.keys(), metric_values))
# for name in names_to_values:
#     print('%s: %f' % (name, names_to_values[name]))
