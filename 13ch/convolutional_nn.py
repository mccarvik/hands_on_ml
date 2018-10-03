import os, pdb, time, sys, time
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import load_sample_image
# from PIL.Image import core as _imaging
# from PIL.Image import core as _imaging
# import  PIL
# from PIL import Image
# from PIL import _imaging
PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/13ch/'


def mnist_ex():
    height = 28
    width = 28
    channels = 1
    n_inputs = height * width
    
    conv1_fmaps = 32
    conv1_ksize = 3
    conv1_stride = 1
    conv1_pad = "SAME"
    
    conv2_fmaps = 64
    conv2_ksize = 3
    conv2_stride = 2
    conv2_pad = "SAME"
    pool3_fmaps = conv2_fmaps
    
    n_fc1 = 64
    n_outputs = 10
    reset_graph()
    with tf.name_scope("inputs"):
        X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
        X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
        y = tf.placeholder(tf.int32, shape=[None], name="y")
    
    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv1")
    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                             strides=conv2_stride, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv2")
    
    with tf.name_scope("pool3"):
        pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])
    
    with tf.name_scope("fc1"):
        fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
    
    with tf.name_scope("output"):
        logits = tf.layers.dense(fc1, n_outputs, name="output")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")
    
    with tf.name_scope("train"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)
    
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    mnist = input_data.read_data_sets("/tmp/data/")
    n_epochs = 11
    batch_size = 10
    pdb.set_trace()
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            # for iteration in range(mnist.train.num_examples // batch_size):
            # chopped this down so it will run, memory issues
            for iteration in range(100 // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: mnist.test.images[:20], y: mnist.test.labels[:20]})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
            # Cant save, too much memory
            # save_path = saver.save(sess, "./my_mnist_model")


def pooling_layer():
    china = load_sample_image("china.jpg")
    flower = load_sample_image("flower.jpg")
    dataset = np.array([china, flower], dtype=np.float32)
    batch_size, height, width, channels = dataset.shape

    filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
    filters[:, 3, :, 0] = 1  # vertical line
    filters[3, :, :, 1] = 1  # horizontal line
    X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")
    
    with tf.Session() as sess:
        output = sess.run(max_pool, feed_dict={X: dataset})
    
    plt.imshow(output[0].astype(np.uint8))  # plot the output for the 1st image
    plot_color_image(dataset[0])
    plt.savefig(PNG_PATH + "china_original2", dpi=300)
    plt.close()
    plot_color_image(output[0])
    plt.savefig(PNG_PATH + "china_max_pool", dpi=300)
    plt.close()        


def valid_vs_same_padding():
    reset_graph()
    filter_primes = np.array([2., 3., 5., 7., 11., 13.], dtype=np.float32)
    x = tf.constant(np.arange(1, 13+1, dtype=np.float32).reshape([1, 1, 13, 1]))
    filters = tf.constant(filter_primes.reshape(1, 6, 1, 1))
    
    valid_conv = tf.nn.conv2d(x, filters, strides=[1, 1, 5, 1], padding='VALID')
    same_conv = tf.nn.conv2d(x, filters, strides=[1, 1, 5, 1], padding='SAME')
    
    with tf.Session() as sess:
        print("VALID:\n", valid_conv.eval())
        print("SAME:\n", same_conv.eval())
    
    print("VALID:")
    print(np.array([1,2,3,4,5,6]).T.dot(filter_primes))
    print(np.array([6,7,8,9,10,11]).T.dot(filter_primes))
    print("SAME:")
    print(np.array([0,1,2,3,4,5]).T.dot(filter_primes))
    print(np.array([5,6,7,8,9,10]).T.dot(filter_primes))
    print(np.array([10,11,12,13,0,0]).T.dot(filter_primes))


def simple_example():
    # Load sample images
    china = load_sample_image("china.jpg")
    flower = load_sample_image("flower.jpg")
    dataset = np.array([china, flower], dtype=np.float32)
    batch_size, height, width, channels = dataset.shape
    
    # Create 2 filters
    filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
    filters[:, 3, :, 0] = 1  # vertical line
    filters[3, :, :, 1] = 1  # horizontal line

    # Create a graph with input X plus a convolutional layer applying the 2 filters
    X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")
    
    with tf.Session() as sess:
        output = sess.run(convolution, feed_dict={X: dataset})
    
    plt.imshow(output[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
    plt.show()

    for image_index in (0, 1):
        for feature_map_index in (0, 1):
            plot_image(output[image_index, :, :, feature_map_index])
            plt.savefig(PNG_PATH + "conv_imgs" + str(image_index) + str(feature_map_index), dpi=300)
            plt.close()
    
    # Using tf.layers.conv2d():
    reset_graph()
    X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
    conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2,2], padding="SAME")
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        init.run()
        output = sess.run(conv, feed_dict={X: dataset})
    
    plt.imshow(output[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
    plt.savefig(PNG_PATH + "conv2d", dpi=300)
    plt.close()


def convolutional_layer():
    pdb.set_trace()
    china = load_sample_image("china.jpg")
    flower = load_sample_image("flower.jpg")
    image = china[150:220, 130:250]
    height, width, channels = image.shape
    image_grayscale = image.mean(axis=2).astype(np.float32)
    images = image_grayscale.reshape(1, height, width, 1)
    fmap = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
    fmap[:, 3, 0, 0] = 1
    fmap[3, :, 0, 1] = 1
    plot_image(fmap[:, :, 0, 0])
    plt.savefig(PNG_PATH + "vertical", dpi=300)
    plt.close()
    plot_image(fmap[:, :, 0, 1])
    plt.savefig(PNG_PATH + "horizontal", dpi=300)
    plt.close()
    
    reset_graph()
    X = tf.placeholder(tf.float32, shape=(None, height, width, 1))
    feature_maps = tf.constant(fmap)
    convolution = tf.nn.conv2d(X, feature_maps, strides=[1,1,1,1], padding="SAME")

    with tf.Session() as sess:
        output = convolution.eval(feed_dict={X: images})
    plot_image(images[0, :, :, 0])
    plt.savefig(PNG_PATH + "china_original", dpi=300)
    plt.close()
    
    plot_image(output[0, :, :, 0])
    plt.savefig(PNG_PATH + "china_vertical", dpi=300)
    plt.close()
    
    plot_image(output[0, :, :, 1])
    plt.savefig(PNG_PATH + "china_horizontal", dpi=300)
    plt.close()
    
    
def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")
    
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)    

if __name__ == '__main__':
    # convolutional_layer()
    # simple_example()
    # valid_vs_same_padding()
    # pooling_layer()
    mnist_ex()