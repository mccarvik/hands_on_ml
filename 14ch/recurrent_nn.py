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
from sklearn.manifold import TSNE

from six.moves import urllib
import errno
import os
import zipfile
from collections import Counter, deque

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/14ch/'
WORDS_PATH = "datasets/words"
WORDS_URL = 'http://mattmahoney.net/dc/text8.zip'


def machine_translation():
    # Will use ML to translate english to french
    reset_graph()
    n_steps = 50
    n_neurons = 200
    n_layers = 3
    num_encoder_symbols = 20000
    num_decoder_symbols = 20000
    embedding_size = 150
    learning_rate = 0.01
    
    X = tf.placeholder(tf.int32, [None, n_steps]) # English sentences
    Y = tf.placeholder(tf.int32, [None, n_steps]) # French translations
    W = tf.placeholder(tf.float32, [None, n_steps - 1, 1])
    Y_input = Y[:, :-1]
    Y_target = Y[:, 1:]
    
    encoder_inputs = tf.unstack(tf.transpose(X)) # list of 1D tensors
    decoder_inputs = tf.unstack(tf.transpose(Y_input)) # list of 1D tensors
    
    lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons) for layer in range(n_layers)]
    cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
    
    output_seqs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        encoder_inputs,
        decoder_inputs,
        cell,
        num_encoder_symbols,
        num_decoder_symbols,
        embedding_size)
    
    logits = tf.transpose(tf.unstack(output_seqs), perm=[1, 0, 2])
    logits_flat = tf.reshape(logits, [-1, num_decoder_symbols])
    Y_target_flat = tf.reshape(Y_target, [-1])
    W_flat = tf.reshape(W, [-1])
    xentropy = W_flat * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_target_flat, logits=logits_flat)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()


def embeddings_and_nlp():
    words = fetch_words_data()
    print(words[:5])
    vocabulary_size = 50000

    vocabulary = [("UNK", None)] + Counter(words).most_common(vocabulary_size - 1)
    vocabulary = np.array([word for word, _ in vocabulary])
    dictionary = {word: code for code, word in enumerate(vocabulary)}
    data = np.array([dictionary.get(word, 0) for word in words])
    print(" ".join(words[:9]), data[:9])
    print(" ".join([vocabulary[word_index] for word_index in [5241, 3081, 12, 6, 195, 2, 3134, 46, 59]]))
    print(words[24], data[24])
    np.random.seed(42)
    data_index = 0
    
    def generate_batch(batch_size, num_skips, skip_window, data_index):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=[batch_size], dtype=np.int32)
        labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        buffer = deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = np.random.randint(0, span)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        return batch, labels, data_index
    
    batch, labels, data_index = generate_batch(8, 2, 1, data_index)
    print(batch, [vocabulary[word] for word in batch])
    print(labels, [vocabulary[word] for word in labels[:, 0]])
    
    # build the model
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.
    
    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64    # Number of negative examples to sample.
    
    learning_rate = 0.01
    reset_graph()
    
    # Input data.
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    vocabulary_size = 50000
    embedding_size = 150
    
    # Look up embeddings for inputs.
    init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
    embeddings = tf.Variable(init_embeds)
    train_inputs = tf.placeholder(tf.int32, shape=[None])
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed, num_sampled, vocabulary_size))
    
    # Construct the Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
    # Add variable initializer.
    init = tf.global_variables_initializer()
    
    # Train the model
    num_steps = 100
    with tf.Session() as session:
        init.run()
        average_loss = 0
        for step in range(num_steps):
            print("\rIteration: {}".format(step), end="\t")
            batch_inputs, batch_labels, data_index = generate_batch(batch_size, num_skips, skip_window, data_index)
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
    
            # We perform one update step by evaluating the training op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([training_op, loss], feed_dict=feed_dict)
            average_loss += loss_val
    
            if step % 20 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
    
            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 100 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = vocabulary[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = vocabulary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()
    
    # np.save("./my_final_embeddings.npy", final_embeddings)
    # Plot embeddings
    pdb.set_trace()
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
    labels = [vocabulary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)
    plt.savefig(PNG_PATH + "embeddings", dpi=300)
    plt.close()


def lstm():
    reset_graph()
    n_steps = 28
    n_inputs = 28
    n_neurons = 150
    n_outputs = 10
    n_layers = 3
    learning_rate = 0.001
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])
    lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons) for layer in range(n_layers)]
    multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
    outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
    top_layer_h_state = states[-1][1]
    logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
    init = tf.global_variables_initializer()
    print(states)
    print(top_layer_h_state)
    
    n_epochs = 11
    batch_size = 10
    mnist = input_data.read_data_sets("/tmp/data/")
    X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
    y_test = mnist.test.labels
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(200 // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch = X_batch.reshape((batch_size, n_steps, n_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print("Epoch", epoch, "Train accuracy =", acc_train, "Test accuracy =", acc_test)
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, use_peepholes=True)
    gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)


def dropout():
    reset_graph()
    n_inputs = 1
    n_neurons = 100
    n_layers = 3
    n_steps = 20
    n_outputs = 1
    resolution = 0.1
    t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    
    keep_prob = tf.placeholder_with_default(1.0, shape=())
    cells = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) for layer in range(n_layers)]
    cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in cells]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
    rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    learning_rate = 0.01
    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    n_iterations = 800
    batch_size = 50
    train_keep_prob = 0.5
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            X_batch, y_batch = next_batch(batch_size, n_steps)
            _, mse = sess.run([training_op, loss], feed_dict={X: X_batch, y: y_batch, keep_prob: train_keep_prob})
            if iteration % 100 == 0:                   # not shown in the book
                print(iteration, "Training MSE:", mse) # not shown
        saver.save(sess, "./my_dropout_time_series_model")

    with tf.Session() as sess:
        saver.restore(sess, "./my_dropout_time_series_model")
    
        X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
        y_pred = sess.run(outputs, feed_dict={X: X_new})
        plt.title("Testing the model", fontsize=14)
        plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
        plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
        plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
        plt.legend(loc="upper left")
        plt.xlabel("Time")
        plt.savefig(PNG_PATH + "dropout", dpi=300)
        plt.close()


def distribute_deep_rnn_across_gpu():
    # dont do this:
    # with tf.device("/gpu:0"):  # BAD! This is ignored.
    #     layer1 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

    # with tf.device("/gpu:1"):  # BAD! Ignored again.
    #     layer2 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    reset_graph()
    n_inputs = 5
    n_steps = 20
    n_neurons = 100
    
    X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs])
    devices = ["/cpu:0", "/cpu:0", "/cpu:0"] # replace with ["/gpu:0", "/gpu:1", "/gpu:2"] if you have 3 GPUs
    cells = [DeviceCellWrapper(dev,tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)) for dev in devices]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        print(sess.run(outputs, feed_dict={X: np.random.rand(2, n_steps, n_inputs)}))
    

def multi_rnn_cell():
    reset_graph()
    n_inputs = 2
    n_steps = 5
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    n_neurons = 100
    n_layers = 3
    
    layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
    init = tf.global_variables_initializer()
    X_batch = np.random.rand(2, n_steps, n_inputs)
    with tf.Session() as sess:
        init.run()
        outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch})
    print(outputs_val.shape)


def without_OPW():
    reset_graph()
    n_steps = 20
    n_inputs = 1
    n_outputs = 1
    n_neurons = 100
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    n_outputs = 1
    learning_rate = 0.001
    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_iterations = 800
    batch_size = 50
    t_min, t_max = 0, 30
    resolution = 0.1
    t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)
    t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            X_batch, y_batch = next_batch(batch_size, n_steps)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)
        
        X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
        y_pred = sess.run(outputs, feed_dict={X: X_new})
        saver.save(sess, "./my_time_series_model")
        
        print(y_pred)
        plt.title("Testing the model", fontsize=14)
        plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
        plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
        plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
        plt.legend(loc="upper left")
        plt.xlabel("Time")
        plt.savefig(PNG_PATH + "time_series_plot_proj_ts", dpi=300)
        plt.close()
    
    
    with tf.Session() as sess: 
        # not shown in the book
        saver.restore(sess, "./my_time_series_model") # not shown
        sequence = [0.] * n_steps
        for iteration in range(300):
            X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, 1)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
            sequence.append(y_pred[0, -1, 0])
            
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(len(sequence)), sequence, "b-")
    plt.plot(t[:n_steps], sequence[:n_steps], "b-", linewidth=3)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.savefig(PNG_PATH + "initial_seq_creative_sequence_plot", dpi=300)
    plt.close()
    
    with tf.Session() as sess:
        saver.restore(sess, "./my_time_series_model")
    
        sequence1 = [0. for i in range(n_steps)]
        for iteration in range(len(t) - n_steps):
            X_batch = np.array(sequence1[-n_steps:]).reshape(1, n_steps, 1)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
            sequence1.append(y_pred[0, -1, 0])
    
        sequence2 = [time_series(i * resolution + t_min + (t_max-t_min/3)) for i in range(n_steps)]
        for iteration in range(len(t) - n_steps):
            X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
            sequence2.append(y_pred[0, -1, 0])
    
    plt.figure(figsize=(11,4))
    plt.subplot(121)
    plt.plot(t, sequence1, "b-")
    plt.plot(t[:n_steps], sequence1[:n_steps], "b-", linewidth=3)
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    plt.subplot(122)
    plt.plot(t, sequence2, "b-")
    plt.plot(t[:n_steps], sequence2[:n_steps], "b-", linewidth=3)
    plt.xlabel("Time")
    plt.savefig(PNG_PATH + "creative_sequence_plot", dpi=300)
    plt.close()


def using_output_projection_wrapper():
    reset_graph()
    n_steps = 20
    n_inputs = 1
    n_neurons = 100
    n_outputs = 1
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    
    cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    
    reset_graph()
    n_steps = 20
    n_inputs = 1
    n_neurons = 100
    n_outputs = 1
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    learning_rate = 0.001
    loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    n_iterations = 800
    batch_size = 50
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            X_batch, y_batch = next_batch(batch_size, n_steps)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)
        saver.save(sess, "./my_time_series_model") # not shown in the book
    
    with tf.Session() as sess:                          # not shown in the book
        saver.restore(sess, "./my_time_series_model")   # not shown
        resolution = 0.1
        t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)
        X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
        y_pred = sess.run(outputs, feed_dict={X: X_new})
        print(y_pred)
        
        plt.title("Testing the model", fontsize=14)
        plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
        plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
        plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
        plt.legend(loc="upper left")
        plt.xlabel("Time")
        plt.savefig(PNG_PATH + "time_series_pred_plot", dpi=300)
        plt.close()


def time_series_projection():
    t_min, t_max = 0, 30
    resolution = 0.1
    t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))
    n_steps = 20
    t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)
    
    plt.figure(figsize=(11,4))
    plt.subplot(121)
    plt.title("A time series (generated)", fontsize=14)
    plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
    plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
    plt.legend(loc="lower left", fontsize=14)
    plt.axis([0, 30, -17, 13])
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    plt.subplot(122)
    plt.title("A training instance", fontsize=14)
    plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
    plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
    plt.legend(loc="upper left")
    plt.xlabel("Time")
    
    
    plt.savefig(PNG_PATH + "time_series_plot", dpi=300)
    plt.close()

    X_batch, y_batch = next_batch(1, n_steps)
    print(np.c_[X_batch[0], y_batch[0]])


def multi_layer_rnn():
    reset_graph()
    n_steps = 28
    n_inputs = 28
    n_outputs = 10
    
    learning_rate = 0.001
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])

    n_neurons = 100
    n_layers = 3
    layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    states_concat = tf.concat(axis=1, values=states)
    logits = tf.layers.dense(states_concat, n_outputs)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    init = tf.global_variables_initializer()
    n_epochs = 11
    batch_size = 10
    mnist = input_data.read_data_sets("/tmp/data/")
    X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
    y_test = mnist.test.labels
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(100 // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
            

def seq_classifier():
    reset_graph()
    n_steps = 28
    n_inputs = 28
    n_neurons = 150
    n_outputs = 10
    
    learning_rate = 0.001
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])
    
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
    
    logits = tf.layers.dense(states, n_outputs)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    init = tf.global_variables_initializer()
    mnist = input_data.read_data_sets("/tmp/data/")
    X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
    y_test = mnist.test.labels
    
    n_epochs = 11
    batch_size = 150
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)


def setting_sequence_lens():
    n_steps = 2
    n_inputs = 3
    n_neurons = 5
    reset_graph()
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    seq_length = tf.placeholder(tf.int32, [None])
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,
                                        sequence_length=seq_length)
    init = tf.global_variables_initializer()
    X_batch = np.array([
            # step 0     step 1
            [[0, 1, 2], [9, 8, 7]], # instance 1
            [[3, 4, 5], [0, 0, 0]], # instance 2 (padded with zero vectors)
            [[6, 7, 8], [6, 5, 4]], # instance 3
            [[9, 0, 1], [3, 2, 1]], # instance 4
        ])
    # define the seq length for eavh entry
    seq_length_batch = np.array([2, 1, 2, 2])

    with tf.Session() as sess:
        init.run()
        outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})
    print(outputs_val)
    print(states_val)


def using_dynamic_rnn():
    n_steps = 2
    n_inputs = 3
    n_neurons = 5
    reset_graph()
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
    init = tf.global_variables_initializer()
    X_batch = np.array([
            [[0, 1, 2], [9, 8, 7]], # instance 1
            [[3, 4, 5], [0, 0, 0]], # instance 2
            [[6, 7, 8], [6, 5, 4]], # instance 3
            [[9, 0, 1], [3, 2, 1]], # instance 4
        ])
    
    with tf.Session() as sess:
        init.run()
        outputs_val = outputs.eval(feed_dict={X: X_batch})
    print(outputs_val)


def packing_sequences():
    n_steps = 2
    n_inputs = 3
    n_neurons = 5
    reset_graph()
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
    
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
    outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])
    init = tf.global_variables_initializer()
    X_batch = np.array([
            # t = 0      t = 1 
            [[0, 1, 2], [9, 8, 7]], # instance 1
            [[3, 4, 5], [0, 0, 0]], # instance 2
            [[6, 7, 8], [6, 5, 4]], # instance 3
            [[9, 0, 1], [3, 2, 1]], # instance 4
        ])
    
    with tf.Session() as sess:
        init.run()
        outputs_val = outputs.eval(feed_dict={X: X_batch})
    print(outputs_val)
    print(np.transpose(outputs_val, axes=[1, 0, 2])[1])


def using_static_rnn():
    n_inputs = 3
    n_neurons = 5
    reset_graph()
    
    X0 = tf.placeholder(tf.float32, [None, n_inputs])
    X1 = tf.placeholder(tf.float32, [None, n_inputs])
    
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
    Y0, Y1 = output_seqs
    init = tf.global_variables_initializer()
    X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
    X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])
    
    with tf.Session() as sess:
        init.run()
        Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

    print(Y0_val)
    print(Y1_val)


def basic_rnns():
    reset_graph()
    n_inputs = 3
    n_neurons = 5
    
    X0 = tf.placeholder(tf.float32, [None, n_inputs])
    X1 = tf.placeholder(tf.float32, [None, n_inputs])
    
    Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
    Wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
    b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))
    
    Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
    Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

    init = tf.global_variables_initializer()
    X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
    X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1
    
    with tf.Session() as sess:
        init.run()
        Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
    print(Y0_val)
    print(Y1_val)


class DeviceCellWrapper(tf.contrib.rnn.RNNCell):
  def __init__(self, device, cell):
    self._cell = cell
    self._device = device

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    with tf.device(self._device):
        return self._cell(inputs, state, scope)

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)
    
def next_batch(batch_size, n_steps):
    t_min, t_max = 0, 30
    resolution = 0.1
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)    

def mkdir_p(path):
    """Create directories, ok if they already exist.
    
    This is for python 2 support. In python >=3.2, simply use:
    >>> os.makedirs(path, exist_ok=True)
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def fetch_words_data(words_url=WORDS_URL, words_path=WORDS_PATH):
    os.makedirs(words_path, exist_ok=True)
    # zip_path = os.path.join(words_path, "words.zip")
    # Words_short may have been deleted to not send to git
    zip_path = os.path.join(words_path, "words_short.zip")
    if not os.path.exists(zip_path):
        # dont want to actually do this, file too big
        return
        urllib.request.urlretrieve(words_url, zip_path)
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    return data.decode("ascii").split()

def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

if __name__ == '__main__':
    # basic_rnns()
    # using_static_rnn()
    # packing_sequences()
    # using_dynamic_rnn()
    # setting_sequence_lens()
    # seq_classifier()
    # multi_layer_rnn()
    # time_series_projection()
    # using_output_projection_wrapper()
    # without_OPW()
    # multi_rnn_cell()
    # distribute_deep_rnn_across_gpu()
    # dropout()
    # lstm
    embeddings_and_nlp()
    # machine_translation()