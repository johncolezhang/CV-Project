from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops import *
from data import *
from net import *
from utils import *
import os
import time
import data
from restore import readfile


flags = tf.app.flags
conf = flags.FLAGS
class Solver(object):
  def __init__(self):
    self.device_id = conf.device_id
    self.train_dir = conf.train_dir
    self.samples_dir = conf.samples_dir
    if not os.path.exists(self.train_dir):
      os.makedirs(self.train_dir)
    if not os.path.exists(self.samples_dir):
      os.makedirs(self.samples_dir)    
    #datasets params
    self.num_epoch = conf.num_epoch
    self.batch_size = conf.batch_size
    #optimizer parameter
    self.learning_rate = conf.learning_rate
    if conf.use_gpu:
      device_str = '/gpu:' + str(self.device_id)
    else:
      device_str = '/cpu:0'
    with tf.device(device_str):
      #dataset
      self.dataset = DataSet(conf.imgs_list_path, self.num_epoch, self.batch_size)
      self.net = Net(self.dataset.hr_images, self.dataset.lr_images, 'prsr')
      #optimizer
      self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
      # decay every 500000 steps with a base of 0.5
      learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                           500000, 0.5, staircase=True)
      optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9, epsilon=1e-8)
      self.train_op = optimizer.minimize(self.net.loss, global_step=self.global_step)


  def train(self):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    # Create a session for running operations in the Graph.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize the variables (like the epoch counter).
    sess.run(init_op)
    saver.restore(sess, "../models/model.ckpt-30000")
    summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    iters = 0
    try:
      while not coord.should_stop():
        # Run training steps or whatever
        t1 = time.time()
        _, loss = sess.run([self.train_op, self.net.loss], feed_dict={self.net.train: True})
        t2 = time.time()
        print('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % ((iters, loss, self.batch_size/(t2-t1), (t2-t1))))
        # self.sample(sess, mu=1.1, step=iters)
        iters += 1
        if iters % 10 == 0:
          summary_str = sess.run(summary_op, feed_dict={self.net.train: True})
          summary_writer.add_summary(summary_str, iters)
        if iters % 1 == 0:
          #self.sample(sess, mu=1.0, step=iters)
          self.sample(sess, mu=1.1, step=iters)
          #self.sample(sess, mu=100, step=iters)
        if iters % 10000 == 0:
          checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=iters)
    except tf.errors.OutOfRangeError:
      checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
      saver.save(sess, checkpoint_path)
      print('Done training -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

  def predict(self):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()
    # Create a session for running operations in the Graph.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)

    saver.restore(sess, "../models/model.ckpt-30000")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      self.sample(sess, mu=1.1, step=0)
    except tf.errors.OutOfRangeError:
      checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
      saver.save(sess, checkpoint_path)
      print('Done training -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Start input enqueue threads.
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    # t1 = time.time()
    # _, loss = sess.run([self.train_op, self.net.loss], feed_dict={self.net.train: True})
    # t2 = time.time()
    # print('loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % (loss, self.batch_size / (t2 - t1), (t2 - t1)))

    # image_path = "../data/000063.jpg"
    # filename_queue = tf.train.string_input_producer([image_path])
    # reader = tf.WholeFileReader()
    # key, value = reader.read(filename_queue)
    # my_img = tf.image.decode_jpeg(value, 3)
    # hr_imgs = tf.image.resize_images(my_img, [32, 32])
    # lr_imgs = tf.image.resize_images(my_img, [8, 8])
    # hr_imgs = tf.cast(hr_imgs, tf.float32)
    # lr_imgs = tf.cast(lr_imgs, tf.float32)


    # np_lr_imgs = mpimg.imread(image_path)
    # np_lr_imgs = np_lr_imgs.reshape([32, 32, 3])
    # np_lr_imgs = np.resize(np_lr_imgs, [8, 8, 3]).astype(np.float32)
    #
    # np_lr = []
    # for _ in range(32):
    #   np_lr.append(np_lr_imgs)
    # np_lr_imgs = np.array(np_lr)

    # min_after_dequeue = 1000
    # capacity = min_after_dequeue + 400 * 32
    # hr_imgs, lr_imgs = tf.train.shuffle_batch([hr_imgs, lr_imgs],
    #                                           batch_size=32, capacity=capacity,
    #                                           min_after_dequeue=min_after_dequeue)
    #
    # np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs])
    #
    # # hr_imgs, lr_imgs, np_hr_imgs, np_lr_imgs = readfile()
    #
    #
    # c_logits = self.net.conditioning_logits
    # p_logits = self.net.prior_logits
    #
    # np_c_logits = sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, self.train: False})
    # gen_hr_imgs = np.zeros((1, 32, 32, 3), dtype=np.float32)
    # for i in range(32):
    #   for j in range(32):
    #     for c in range(3):
    #       np_p_logits = sess.run(p_logits, feed_dict={hr_imgs: gen_hr_imgs})
    #       new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256] + np_p_logits[:, i, j, c*256:(c+1)*256], mu=1.1)
    #       gen_hr_imgs[:, i, j, c] = new_pixel
    #
    # coord.join(threads)
    # sess.close()




  def sample(self, sess, mu=1.1, step=None):
    c_logits = self.net.conditioning_logits
    p_logits = self.net.prior_logits
    lr_imgs = self.dataset.lr_images
    hr_imgs = self.dataset.hr_images
    np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs])
    gen_hr_imgs = np.zeros((self.batch_size, 32, 32, 3), dtype=np.float32)
    #gen_hr_imgs = np_hr_imgs
    #gen_hr_imgs[:,16:,16:,:] = 0.0
    np_c_logits = sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, self.train:False})
    print('iters %d: ' % step)
    
    for i in range(32):
      for j in range(32):
        for c in range(3):
          np_p_logits = sess.run(p_logits, feed_dict={hr_imgs: gen_hr_imgs})
          new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256] + np_p_logits[:, i, j, c*256:(c+1)*256], mu=mu)
          gen_hr_imgs[:, i, j, c] = new_pixel
    #
    save_samples(np_lr_imgs, self.samples_dir + '/lr_' + str(mu*10) + '_' + str(step) + '.jpg')
    save_samples(np_hr_imgs, self.samples_dir + '/hr_' + str(mu*10) + '_' + str(step) + '.jpg')
    save_samples(gen_hr_imgs, self.samples_dir + '/generate_' + str(mu*10) + '_' + str(step) + '.jpg')

