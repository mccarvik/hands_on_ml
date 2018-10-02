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

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/11ch/'


def learning_rate_scheduling():
    reset_graph()
    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 50
    n_outputs = 10
    
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")
    
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    with tf.name_scope("train"):       # not shown in the book
        initial_learning_rate = 0.1
        decay_steps = 10000
        decay_rate = 1/10
        global_step = tf.Variable(0, trainable=False, name="global_step")
        # learning rate decreases as training progresses
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        training_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:50], X_train[50:]
    y_valid, y_train = y_train[:50], y_train[50:]
    
    n_epochs = 5
    batch_size = 50
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)
        save_path = saver.save(sess, "./my_model_final.ckpt")


def faster_optimizers():
    # momentum optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    # Nesterov Accelerated Gradient
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9, use_nesterov=True)
    # AdaGrad
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    # RMSProp
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9, decay=0.9, epsilon=1e-10)
    # Adam Optimization
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


def cached_frozen_layers():
    reset_graph()

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300 # reused
    n_hidden2 = 50  # reused
    n_hidden3 = 50  # reused
    n_hidden4 = 20  # new!
    n_outputs = 10  # new!

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")       # reused
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2") # reused
        # Can add a stop gradient to freeze the first two layers instead of using the method shown
        # hidden2_stop = tf.stop_gradient(hidden2)
        hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3") # reused
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4") # new!
        logits = tf.layers.dense(hidden4, n_outputs, name="outputs")                         # new!

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    learning_rate = 0.01
    with tf.name_scope("train"):   # not shown in the book
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)     # not shown
        # only traing the 3rd and 4th layer
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[34]|outputs")
        training_op = optimizer.minimize(loss, var_list=train_vars)

    init = tf.global_variables_initializer()
    new_saver = tf.train.Saver()

    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden[123]") # regular expression
    restore_saver = tf.train.Saver(reuse_vars) # to restore layers 1-3
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:50], X_train[50:]
    y_valid, y_train = y_train[:50], y_train[50:]

    n_epochs = 11
    batch_size = 200
    n_batches = len(X_train) // batch_size

    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, "./my_model_final.ckpt")
        
        h2_cache = sess.run(hidden2, feed_dict={X: X_train})
        h2_cache_valid = sess.run(hidden2, feed_dict={X: X_valid}) # not shown in the book
    
        for epoch in range(n_epochs):
            shuffled_idx = np.random.permutation(len(X_train))
            hidden2_batches = np.array_split(h2_cache[shuffled_idx], n_batches)
            y_batches = np.array_split(y_train[shuffled_idx], n_batches)
            for hidden2_batch, y_batch in zip(hidden2_batches, y_batches):
                sess.run(training_op, feed_dict={hidden2:hidden2_batch, y:y_batch})
            accuracy_val = accuracy.eval(feed_dict={hidden2: h2_cache_valid, # not shown
                                                    y: y_valid})             # not shown
            print(epoch, "Validation accuracy:", accuracy_val)               # not shown
        save_path = saver.save(sess, "./my_new_model_final.ckpt")


def freeze_lower_layers():
    reset_graph()

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300 # reused
    n_hidden2 = 50  # reused
    n_hidden3 = 50  # reused
    n_hidden4 = 20  # new!
    n_outputs = 10  # new!

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")       # reused
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2") # reused
        # Can add a stop gradient to freeze the first two layers instead of using the method shown
        # hidden2_stop = tf.stop_gradient(hidden2)
        hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3") # reused
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4") # new!
        logits = tf.layers.dense(hidden4, n_outputs, name="outputs")                         # new!

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    learning_rate = 0.01
    with tf.name_scope("train"):   # not shown in the book
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)     # not shown
        # only traing the 3rd and 4th layer
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[34]|outputs")
        training_op = optimizer.minimize(loss, var_list=train_vars)

    init = tf.global_variables_initializer()
    new_saver = tf.train.Saver()

    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden[123]") # regular expression
    restore_saver = tf.train.Saver(reuse_vars) # to restore layers 1-3
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:50], X_train[50:]
    y_valid, y_train = y_train[:50], y_train[50:]

    n_epochs = 11
    batch_size = 200
    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, "./my_model_final.ckpt")
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)
        save_path = saver.save(sess, "./my_new_model_final.ckpt")


def reuse_other_framework():
    reset_graph()
    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300

    original_w = [[1., 2., 3.], [4., 5., 6.]] # Load the weights from the other framework
    original_b = [7., 8., 9.]                 # Load the biases from the other framework
    
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    # [...] Build the rest of the model
    
    # Get a handle on the assignment nodes for the hidden1 variables
    graph = tf.get_default_graph()
    assign_kernel = graph.get_operation_by_name("hidden1/kernel/Assign")
    assign_bias = graph.get_operation_by_name("hidden1/bias/Assign")
    init_kernel = assign_kernel.inputs[1]
    init_bias = assign_bias.inputs[1]
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init, feed_dict={init_kernel: original_w, init_bias: original_b})
        # [...] Train the model on your new task
        print(hidden1.eval(feed_dict={X: [[10.0, 11.0]]}))  # not shown in the book
    

def reuse_part_of_model():
    reset_graph()
    n_hidden4 = 20  # new layer
    n_outputs = 10  # new layer

    saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")
    
    X = tf.get_default_graph().get_tensor_by_name("X:0")
    y = tf.get_default_graph().get_tensor_by_name("y:0")
    
    hidden3 = tf.get_default_graph().get_tensor_by_name("dnn/hidden3/Relu:0")
    
    # adding 4th hidden layer on top of 3 we loaded
    new_hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="new_hidden4")
    new_logits = tf.layers.dense(new_hidden4, n_outputs, name="new_outputs")
    
    with tf.name_scope("new_loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    
    with tf.name_scope("new_eval"):
        correct = tf.nn.in_top_k(new_logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    
    learning_rate = 0.01
    with tf.name_scope("new_train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    new_saver = tf.train.Saver()
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:50], X_train[50:]
    y_valid, y_train = y_train[:50], y_train[50:]

    n_epochs = 11
    batch_size = 200
    with tf.Session() as sess:
        init.run()
        saver.restore(sess, "./my_model_final.ckpt")
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)
        save_path = new_saver.save(sess, "./my_new_model_final.ckpt")


def reuse_tf_model():
    reset_graph()
    saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")
    for op in tf.get_default_graph().get_operations():
        print(op.name)
    
    X = tf.get_default_graph().get_tensor_by_name("X:0")
    y = tf.get_default_graph().get_tensor_by_name("y:0")

    accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")
    training_op = tf.get_default_graph().get_operation_by_name("GradientDescent")
    
    for op in (X, y, accuracy, training_op):
        tf.add_to_collection("my_important_ops", op)

    X, y, accuracy, training_op = tf.get_collection("my_important_ops")
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")

    # continue training the model...
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:50], X_train[50:]
    y_valid, y_train = y_train[:50], y_train[50:]

    n_epochs = 11
    batch_size = 200
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)
        save_path = saver.save(sess, "./my_new_model_final.ckpt")


def gradient_clipping():
    # Gradient clipping --> clip the gradients during back propigation so that they never exceed some threshold
    reset_graph()
    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 50
    n_hidden3 = 50
    n_hidden4 = 50
    n_hidden5 = 50
    n_outputs = 10
    
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
        hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
        hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
        logits = tf.layers.dense(hidden5, n_outputs, name="outputs")
    
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01
    threshold = 1.0

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
    training_op = optimizer.apply_gradients(capped_gvs)
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:50], X_train[50:]
    y_valid, y_train = y_train[:50], y_train[50:]

    n_epochs = 11
    batch_size = 200
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)
        save_path = saver.save(sess, "./my_model_final.ckpt")


def batch_normalization():
    reset_graph()
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    
    reset_graph()
    batch_norm_momentum = 0.9
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")
    training = tf.placeholder_with_default(False, shape=(), name='training')
    
    with tf.name_scope("dnn"):
        he_init = tf.variance_scaling_initializer()
        my_batch_norm_layer = partial( tf.layers.batch_normalization, training=training, momentum=batch_norm_momentum)
        my_dense_layer = partial(tf.layers.dense, kernel_initializer=he_init)
        hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
        bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
        hidden2 = my_dense_layer(bn1, n_hidden2, name="hidden2")
        bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
        logits_before_bn = my_dense_layer(bn2, n_outputs, name="outputs")
        logits = my_batch_norm_layer(logits_before_bn)
    
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    
    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:50], X_train[50:]
    y_valid, y_train = y_train[:50], y_train[50:]
    
    n_epochs = 11
    batch_size = 200
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run([training_op, extra_update_ops],
                         feed_dict={training: True, X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)
        save_path = saver.save(sess, "./my_model_final.ckpt")
    print([v.name for v in tf.trainable_variables()])
    print([v.name for v in tf.global_variables()])


def elu_act_mnist():
    reset_graph()
    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        # for selu activation instead of leaky_relu
        # hidden1 = tf.layers.dense(X, n_hidden1, activation=selu, name="hidden1")
        # hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=selu, name="hidden2")
        hidden1 = tf.layers.dense(X, n_hidden1, activation=selu_new, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=selu_new, name="hidden2")
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:50], X_train[50:]
    y_valid, y_train = y_train[:50], y_train[50:]

    n_epochs = 10
    batch_size = 50
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if epoch % 5 == 0:
                acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
                print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)
        save_path = saver.save(sess, "./my_model_final.ckpt")


def selu_graph():
    z = np.linspace(-5, 5, 200)
    plt.plot(z, selu(z), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [-1.758, -1.758], 'k--')
    plt.plot([0, 0], [-2.2, 3.2], 'k-')
    plt.grid(True)
    plt.title(r"SELU activation function", fontsize=14)
    plt.axis([-5, 5, -2.2, 3.2])
    plt.savefig(PNG_PATH + "selu_plot", dpi=300)
    plt.close()


def exponential_linear_unit_graph():
    z = np.linspace(-5, 5, 200)
    plt.plot(z, elu(z), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [-1, -1], 'k--')
    plt.plot([0, 0], [-2.2, 3.2], 'k-')
    plt.grid(True)
    plt.title(r"ELU activation function ($\alpha=1$)", fontsize=14)
    plt.axis([-5, 5, -2.2, 3.2])
    plt.savefig(PNG_PATH + "elu_plot", dpi=300)
    plt.close()
    
    # Using ELU with tensorflow
    # reset_graph()
    # X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    # hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name="hidden1")


def leaky_relu_graph():
    z = np.linspace(-5, 5, 200)
    plt.plot(z, leaky_relu(z, 0.05), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([0, 0], [-0.5, 4.2], 'k-')
    plt.grid(True)
    props = dict(facecolor='black', shrink=0.1)
    plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
    plt.title("Leaky ReLU activation function", fontsize=14)
    plt.axis([-5, 5, -0.5, 4.2])
    plt.savefig(PNG_PATH + "leaky_relu_plot", dpi=300)
    plt.close()


def saturation():
    # at 1 and 0 the function saturates, little left for the lower layers to learn
    z = np.linspace(-5, 5, 200)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [1, 1], 'k--')
    plt.plot([0, 0], [-0.2, 1.2], 'k-')
    plt.plot([-5, 5], [-3/4, 7/4], 'g--')
    plt.plot(z, logit(z), "b-", linewidth=2)
    props = dict(facecolor='black', shrink=0.1)
    plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
    plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
    plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")
    plt.grid(True)
    plt.title("Sigmoid activation function", fontsize=14)
    plt.axis([-5, 5, -0.2, 1.2])
    plt.savefig(PNG_PATH + "sigmoid_saturation_plot", dpi=300)
    plt.close()
    

def xavier_he_initialization():
    reset_graph()
    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    he_init = tf.variance_scaling_initializer()
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, kernel_initializer=he_init, name="hidden1")


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def logit(z):
    return 1 / (1 + np.exp(-z))

def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha*z, z)

def leaky_relu_new(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)

def selu(z, scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    # like elu but maintains variance eliminating a lot of the gradient dexcent vanishing/expoding issues
    return scale * elu(z, alpha)
    
def selu_new(z, scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))


if __name__ == '__main__':
    # saturation()
    # xavier_he_initialization()
    # leaky_relu_graph()
    # exponential_linear_unit_graph()
    # selu_graph()
    # elu_act_mnist()
    # batch_normalization()
    # gradient_clipping()
    # reuse_tf_model()
    # reuse_part_of_model()
    # reuse_other_framework()
    # freeze_lower_layers()
    # cached_frozen_layers()
    # faster_optimizers()
    learning_rate_scheduling()
    