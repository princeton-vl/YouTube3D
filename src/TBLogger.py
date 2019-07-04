'''
TensorBoard logger.
'''

import tensorflow as tf

class TBLogger(object):
    def __init__(self, folder, flush_secs=60):
        self.summary_map = {}
        self.sess = None
        self.writer = tf.summary.FileWriter(folder, flush_secs=flush_secs)
        self.sess = tf.Session()


    def create_scalar(self, name):
        assert name not in self.summary_map.keys()
        var = tf.placeholder(tf.float32, shape=[])
        summ = tf.summary.scalar(name, var)
        self.summary_map[name] = (var, summ)


    def create_image(self, name, max_outputs=20):
        assert name not in self.summary_map.keys()
        var = tf.placeholder(tf.float32, shape=[None, None, None, None])
        summ = tf.summary.image(name, var, max_outputs=max_outputs)
        self.summary_map[name] = (var, summ)


    def create_histogram(self, name):
        assert name not in self.summary_map.keys()
        var = tf.placeholder(tf.float32)
        summ = tf.summary.histogram(name, var)
        self.summary_map[name] = (var, summ)


    def add_value(self, name, value, step):
        var, summ = self.summary_map[name]
        summ_str = self.sess.run(summ, feed_dict={var : value})
        self.writer.add_summary(summ_str, global_step=step)
