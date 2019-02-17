import tensorflow as tf
from load_data import *
import numpy as np
import math


class CBOW(object):
    def __init__(self, u_i_r_p_m_c_tuple_list, word_num, user_num, item_num, comp_num, month_num, city_num, batch_size=128,
                 embed_size=100, window_size=1, num_sampled=64, learning_rate=0.001, sample_ratio=0.1):

        self.u_i_r_p_m_c_tuple_list = u_i_r_p_m_c_tuple_list
        self.batch_size = batch_size
        self.window_size = window_size  # how many words to consider left and right
        self.sample_ratio = sample_ratio

        g = tf.Graph()
        with g.as_default():
            self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
            self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
            self.comp_id = tf.placeholder(dtype=tf.int32, shape=[None], name='comp_id')
            self.month_id = tf.placeholder(dtype=tf.int32, shape=[None], name='month_id')
            self.city_id = tf.placeholder(dtype=tf.int32, shape=[None], name='city_id')
            self.word_id = tf.placeholder(dtype=tf.int32, shape=[None, self.window_size * 2], name='word_id')
            self.target = tf.placeholder(dtype=tf.int32, shape=[None], name='target')

            #with tf.device(device_name_or_function='/cpu:0'):
            self.word_embeddings = tf.get_variable(name='word_embeddings', shape=[word_num, embed_size], initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.user_embeddings = tf.get_variable(name='user_embeddings', shape=[user_num, embed_size], initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.item_embeddings = tf.get_variable(name='item_embeddings', shape=[item_num, embed_size], initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.comp_embeddings = tf.get_variable(name='comp_embeddings', shape=[comp_num, embed_size], initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.month_embeddings = tf.get_variable(name='month_embeddings', shape=[month_num, embed_size], initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.city_embeddings = tf.get_variable(name='city_embeddings', shape=[city_num, embed_size], initializer=tf.random_uniform_initializer(-1.0, 1.0))

            word_feature = tf.nn.embedding_lookup(params=self.word_embeddings, ids=self.word_id)  # (batch_size, window_size * 2, embed_size)
            user_feature = tf.nn.embedding_lookup(params=self.user_embeddings, ids=tf.expand_dims(self.user_id, -1))  # (batch_size, 1, embed_size)
            item_feature = tf.nn.embedding_lookup(params=self.item_embeddings, ids=tf.expand_dims(self.item_id, -1))
            comp_feature = tf.nn.embedding_lookup(params=self.comp_embeddings, ids=tf.expand_dims(self.comp_id, -1))
            month_feature = tf.nn.embedding_lookup(params=self.month_embeddings, ids=tf.expand_dims(self.month_id, -1))
            city_feature = tf.nn.embedding_lookup(params=self.city_embeddings, ids=tf.expand_dims(self.city_id, -1))

            concat_feature = tf.concat(values=[word_feature, user_feature, item_feature, comp_feature, month_feature, city_feature], axis=1)
            train_input = tf.reduce_mean(input_tensor=concat_feature, axis=1)  # (batch_size, embed_size)
            nce_weights = tf.get_variable(name='nce_weights', shape=[word_num, embed_size], initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embed_size)))
            nce_biases = tf.get_variable(name='nce_biases', shape=[word_num], initializer=tf.constant_initializer(0.1))

            # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss
            losses = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=train_input, labels=tf.expand_dims(self.target, -1), num_sampled=num_sampled, num_classes=word_num)
            self.loss = tf.reduce_mean(input_tensor=losses)

            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=g, config=config)
        self.sess.run(init)

    def train_one_epoch(self):
        total_loss = 0
        count = 0

        u_buffer = []
        i_buffer = []
        p_buffer = []
        m_buffer = []
        c_buffer = []
        input_buffer = []
        output_buffer = []
        for (u, i, rev, p, m, c) in self.u_i_r_p_m_c_tuple_list:
            inputs, output = generate_word_pairs(rev, self.window_size, self.sample_ratio)
            length = len(output)
            u_buffer.extend([u] * length)
            i_buffer.extend([i] * length)
            p_buffer.extend([p] * length)
            m_buffer.extend([m] * length)
            c_buffer.extend([c] * length)
            input_buffer.extend(inputs)
            output_buffer.extend(output)

            while len(output_buffer) >= self.batch_size:
                u_train = np.asarray(u_buffer[:self.batch_size], dtype=np.int32)
                i_train = np.asarray(i_buffer[:self.batch_size], dtype=np.int32)
                p_train = np.asarray(p_buffer[:self.batch_size], dtype=np.int32)
                m_train = np.asarray(m_buffer[:self.batch_size], dtype=np.int32)
                c_train = np.asarray(c_buffer[:self.batch_size], dtype=np.int32)
                input_train = np.asarray(input_buffer[:self.batch_size], dtype=np.int32)
                output_train = np.asarray(output_buffer[:self.batch_size], dtype=np.int32)

                u_buffer = u_buffer[self.batch_size:]
                i_buffer = i_buffer[self.batch_size:]
                p_buffer = p_buffer[self.batch_size:]
                m_buffer = m_buffer[self.batch_size:]
                c_buffer = c_buffer[self.batch_size:]
                input_buffer = input_buffer[self.batch_size:]
                output_buffer = output_buffer[self.batch_size:]

                feed_dict = {self.user_id: u_train,
                             self.item_id: i_train,
                             self.comp_id: p_train,
                             self.month_id: m_train,
                             self.city_id: c_train,
                             self.word_id: input_train,
                             self.target: output_train}
                _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                total_loss += loss
                count += 1

        return total_loss / count

    def get_embeddings(self):
        return self.sess.run([self.word_embeddings, self.user_embeddings, self.item_embeddings, self.comp_embeddings, self.month_embeddings, self.city_embeddings])
