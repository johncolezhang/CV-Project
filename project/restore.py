import tensorflow as tf
from func import *


def readfile():
    with tf.Session() as sess1:
        # saver = tf.train.import_meta_graph('models/model.ckpt-30000.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('models/model.ckpt-30000'))
        batch_size = 32
        image_path = "data/000063.jpg"
        filename_queue = tf.train.string_input_producer([image_path])
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        my_img = tf.image.decode_jpeg(value, 3)
        hr_image = tf.image.resize_images(my_img, [32, 32])
        lr_image = tf.image.resize_images(my_img, [8, 8])
        hr_image = tf.cast(hr_image, tf.float32)
        lr_image = tf.cast(lr_image, tf.float32)
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 400 * batch_size
        hr_images, lr_images = tf.train.shuffle_batch([hr_image, lr_image],
                                                      batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
        init = tf.global_variables_initializer()
        sess1.run(init)
        coord1 = tf.train.Coordinator()
        threads1 = tf.train.start_queue_runners(coord=coord1)
        np_hr_images = hr_images.eval()
        np_lr_images = lr_images.eval()
    return hr_images, lr_images, np_hr_images, np_lr_images

    # save_samples(hr_images, '/Users/johncole/Desktop/hr.jpg')
    # save_samples(lr_images, '/Users/johncole/Desktop/lr.jpg')
