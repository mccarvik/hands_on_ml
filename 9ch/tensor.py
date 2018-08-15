import os, pdb, time, sys, time
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/9ch/'


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def create_graph():
    reset_graph()

    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f = x*x*y + y + 2
    print(f)
    
    # creates a session, initializes the variables, and evaluates and then closes
    sess = tf.Session()
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    print(result)
    sess.close()

    # eliminates the need to call sess.run() all the time, also automativally closes the session
    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()
    print(result)

    # prepare an init node for variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # actually initialize all the variables
        init.run()
        result = f.eval()
    print(result)

    # Interactive session --> automatically sets itself as the default session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    init.run()
    result = f.eval()
    print(result)
    sess.close()


if __name__ == '__main__':
    create_graph()