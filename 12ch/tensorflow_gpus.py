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

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/12ch/'

def local_server():
    c = tf.constant("Hello distributed TensorFlow!")
    server = tf.train.Server.create_local_server()
    with tf.Session(server.target) as sess:
        print(sess.run(c))
        

if __name__ == '__main__':
    local_server()