# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import misc
import tensorflow.contrib.slim.nets

plt.style.use('ggplot')
slim = tf.contrib.slim

tfrecord_dir = '/home/itrocks/Git/Tensorflow/dog-breed-classification.tf/raw_data/dog/tfrecord'
# The directory for logging events and checkpoint files.
logdir = '/home/itrocks/Downloads/dog_model/'
checkpoint_file = tf.train.latest_checkpoint(logdir)
# A image size(width and height are same).
image_size = 224
# The number of classes of datasets.
num_classes = 120

with tf.Graph().as_default() as graph:


  test_file = 'demo/0_n02085620_7.jpg'
  test_file = 'demo/1_n02085782_2.jpg'
  test_file = 'demo/1_n02085782_82.jpg'
  test_file = 'demo/2_n02085936_37.jpg'
  #test_file = 'demo/3_n02086079_146.jpg'
  test_file = 'demo/119_n02116738_124.jpg'

  filename_queue = tf.train.string_input_producer([test_file])  # list of files to read
  raw_img = misc.imread(test_file)

  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)
  print(value)

  new_img = tf.image.decode_jpeg(value, channels=3)
  import preprocess.vgg_preprocessing as vgg_preprocessing

  new_img = vgg_preprocessing.preprocess_image(new_img, image_size, image_size, False)
  new_img = tf.expand_dims(new_img, 0)


  print(new_img)


  # 모델 생성
  vgg = tf.contrib.slim.nets.vgg
  with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, end_points = vgg.vgg_16(inputs=new_img, num_classes=num_classes, is_training=False)


  variables_to_restore = slim.get_variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  def restore_fn(sess):
    return saver.restore(sess, checkpoint_file)


  predictions = tf.argmax(logits, 1)

  sv = tf.train.Supervisor(logdir="./", summary_op=None, saver=None, init_fn=restore_fn)

  with sv.managed_session() as sess:
    result_predictions, r_logits, r_new_img = sess.run([predictions, logits, new_img])
    #prediction_name = dataset.labels_to_name[np.asscalar(result_predictions[0])]
    text = 'Prediction: %s' % (result_predictions)
    print(r_logits)
    print(result_predictions)

    #print(r_new_img[0][100])

    """
    for v in tf.global_variables():
      print(v.name)
    """



    img_plot = plt.imshow(raw_img)

    # Set up the plot and hide axes
    plt.title(text)
    img_plot.axes.get_yaxis().set_ticks([])
    img_plot.axes.get_xaxis().set_ticks([])
    plt.show()

  print("= End =")