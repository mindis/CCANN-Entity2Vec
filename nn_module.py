import tensorflow as tf
import numpy as np
import random


class CCANN(object):
    def __init__(self, train_tuple_list, user, item, comp, month, city, embed_size=None, factor_num=10, batch_size=128, learning_rate=0.001):

        self.train_tuple_list = train_tuple_list
        self.batch_size = batch_size
        if embed_size is None:
            embed_size = user.shape[1]

        g = tf.Graph()
        with g.as_default():
            self.user_id = tf.placeholder(tf.int32, [None], name='user_id')
            self.item_id = tf.placeholder(tf.int32, [None], name='item_id')
            self.comp_id = tf.placeholder(tf.int32, [None], name='comp_id')
            self.month_id = tf.placeholder(tf.int32, [None], name='month_id')
            self.city_id = tf.placeholder(tf.int32, [None], name='city_id')
            self.rating = tf.placeholder(tf.float32, [None], name='rating')

            #with tf.device(device_name_or_function='/cpu:0'):
            if isinstance(user, int):
                self.user_embeddings = tf.get_variable('user_embeddings', shape=[user, embed_size], initializer=tf.random_uniform_initializer(-0.1, 0.1))
                self.item_embeddings = tf.get_variable('item_embeddings', shape=[item, embed_size], initializer=tf.random_uniform_initializer(-0.1, 0.1))
                self.comp_embeddings = tf.get_variable('comp_embeddings', shape=[comp, embed_size], initializer=tf.random_uniform_initializer(-0.1, 0.1))
                self.month_embeddings = tf.get_variable('month_embeddings', shape=[month, embed_size], initializer=tf.random_uniform_initializer(-0.1, 0.1))
                self.city_embeddings = tf.get_variable('city_embeddings', shape=[city, embed_size], initializer=tf.random_uniform_initializer(-0.1, 0.1))
            else:
                self.user_embeddings = tf.get_variable('user_embeddings', initializer=user)
                self.item_embeddings = tf.get_variable('item_embeddings', initializer=item)
                self.comp_embeddings = tf.get_variable('comp_embeddings', initializer=comp)
                self.month_embeddings = tf.get_variable('month_embeddings', initializer=month)
                self.city_embeddings = tf.get_variable('city_embeddings', initializer=city)

            user_feature = tf.nn.embedding_lookup(self.user_embeddings, self.user_id)
            item_feature = tf.nn.embedding_lookup(self.item_embeddings, self.item_id)
            comp_feature = tf.nn.embedding_lookup(self.comp_embeddings, self.comp_id)
            month_feature = tf.nn.embedding_lookup(self.month_embeddings, self.month_id)
            city_feature = tf.nn.embedding_lookup(self.city_embeddings, self.city_id)

            def context_pref(fea, ctx):
                concat_fea = tf.concat([fea, ctx], 1)  # (batch_size, embed_size * 2)
                hidden = tf.layers.dense(inputs=concat_fea, units=embed_size, activation=tf.nn.relu,
                                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         bias_initializer=tf.constant_initializer(0.0))  # (batch_size, embed_size)
                return hidden

            with tf.variable_scope('u_p'):
                u_p = context_pref(user_feature, comp_feature)
            with tf.variable_scope('u_m'):
                u_m = context_pref(user_feature, month_feature)
            with tf.variable_scope('u_c'):
                u_c = context_pref(user_feature, city_feature)
            with tf.variable_scope('i_p'):
                i_p = context_pref(item_feature, comp_feature)
            with tf.variable_scope('i_m'):
                i_m = context_pref(item_feature, month_feature)
            with tf.variable_scope('i_c'):
                i_c = context_pref(item_feature, city_feature)

            u_3d_tensor = tf.stack([u_p, u_m, u_c], axis=1)  # (batch_size, 3, embed_size)
            i_3d_tensor = tf.stack([i_p, i_m, i_c], axis=1)

            u_i_3d_tensor = tf.concat([u_3d_tensor, i_3d_tensor], axis=2)  # (batch_size, 3, embed_size * 2)
            u_i_vector = tf.reshape(u_i_3d_tensor, shape=[-1, embed_size * 2])  # (batch_size * 3, embed_size * 2)
            first = tf.layers.dense(inputs=u_i_vector, units=embed_size, activation=tf.nn.relu,
                                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.0))  # (batch_size * 3, embed_size)
            second = tf.layers.dense(inputs=first, units=1, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))  # (batch_size * 3, 1)
            attn = tf.reshape(second, shape=[-1, 3])  # (batch_size, 3)
            attention_score = tf.expand_dims(input=tf.nn.softmax(attn), axis=-1)  # (batch_size, 3, 1)
            u_final_vec = tf.reduce_sum(u_3d_tensor * attention_score, 1)  # (batch_size, embed_size)
            i_final_vec = tf.reduce_sum(i_3d_tensor * attention_score, 1)

            # weights and bias for FM
            FM_w = tf.get_variable('FM_w', initializer=tf.constant(0.0))
            FM_W = tf.get_variable('FM_W', [embed_size * 2, 1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            FM_V = tf.get_variable('FM_V', [embed_size * 2, factor_num], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            # FM
            X = tf.concat([u_final_vec, i_final_vec], 1)  # (batch_size, embed_size * 2)
            linear = tf.reshape(tf.matmul(X, FM_W), [-1])  # (batch_size,)
            bilinear = tf.square(tf.matmul(X, FM_V)) - tf.matmul(tf.square(X), tf.square(FM_V))  # (batch_size, factor_num)
            self.predicted_rating = FM_w + linear + 0.5 * tf.reduce_sum(bilinear, 1)  # (batch_size,)

            self.FM_loss = tf.losses.mean_squared_error(self.rating, self.predicted_rating)
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.FM_loss)

            init = tf.global_variables_initializer()
            # tf.summary.FileWriter('./graphs', tf.get_default_graph())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=g, config=config)
        self.sess.run(init)
        # tf.summary.FileWriter('./graphs', self.sess.graph)

    def train_one_epoch(self):
        sample_num = len(self.train_tuple_list)
        index_list = list(range(sample_num))
        random.shuffle(index_list)

        offset = 0
        while offset < sample_num:
            start = offset
            offset += self.batch_size
            offset = min(offset, sample_num)

            user = []
            item = []
            rating = []
            comp = []
            month = []
            city = []
            for idx in index_list[start:offset]:
                x = self.train_tuple_list[idx]
                user.append(x[0])
                item.append(x[1])
                rating.append(x[2])
                comp.append(x[3])
                month.append(x[4])
                city.append(x[5])
            user = np.asarray(user, dtype=np.int32)
            item = np.asarray(item, dtype=np.int32)
            rating = np.asarray(rating, dtype=np.float32)
            comp = np.asarray(comp, dtype=np.int32)
            month = np.asarray(month, dtype=np.int32)
            city = np.asarray(city, dtype=np.int32)

            feed_dict = {self.user_id: user,
                         self.item_id: item,
                         self.comp_id: comp,
                         self.month_id: month,
                         self.city_id: city,
                         self.rating: rating}
            _, loss = self.sess.run([self.optimizer, self.FM_loss], feed_dict=feed_dict)

    def get_prediction(self, tuple_list):
        sample_num = len(tuple_list)
        prediction = []

        offset = 0
        while offset < sample_num:
            start = offset
            offset += self.batch_size
            offset = min(offset, sample_num)

            user = []
            item = []
            comp = []
            month = []
            city = []
            for x in tuple_list[start:offset]:
                user.append(x[0])
                item.append(x[1])
                comp.append(x[3])
                month.append(x[4])
                city.append(x[5])
            user = np.asarray(user, dtype=np.int32)
            item = np.asarray(item, dtype=np.int32)
            comp = np.asarray(comp, dtype=np.int32)
            month = np.asarray(month, dtype=np.int32)
            city = np.asarray(city, dtype=np.int32)

            feed_dict = {self.user_id: user,
                         self.item_id: item,
                         self.comp_id: comp,
                         self.month_id: month,
                         self.city_id: city}
            predicted_rating = self.sess.run(self.predicted_rating, feed_dict=feed_dict)
            prediction.extend(predicted_rating)

        predicted = [(x[0][2], x[1]) for x in zip(tuple_list, prediction)]

        return predicted

    def get_embeddings(self):
        return self.sess.run([self.user_embeddings, self.item_embeddings, self.comp_embeddings, self.month_embeddings, self.city_embeddings])
