from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import datetime
import random
import sys
import os


def load_raw_data(data_path, max_word_num=20000):
    if not os.path.exists(data_path):
        sys.exit(get_now_time() + 'invalid directory')
    else:
        print(get_now_time() + 'loading raw data')

        doc_list = []
        with open(data_path, 'r', errors='ignore') as f:
            for line in f.readlines():
                content = line.strip().split('::')
                review = content[3]
                doc_list.append(review)

        word2index, index2word = get_word2index(doc_list, max_word_num)

        u_i_r_p_m_c_tuple_list = []
        with open(data_path, 'r', errors='ignore') as f:
            for line in f.readlines():
                content = line.strip().split('::')
                u = content[0]
                i = content[1]
                rev = content[3]
                p = content[4]
                m = content[5]
                c = content[6]

                word_list = []
                for w in rev.split(' '):
                    word_list.append(word2index.get(w, word2index['<UNK>']))
                u_i_r_p_m_c_tuple_list.append((u, i, word_list, p, m, c))

    return u_i_r_p_m_c_tuple_list, index2word


def get_word2index(doc_list, max_word_num=20000):
    def split_words_by_space(text):
        return text.split(' ')

    vectorizer = CountVectorizer(max_features=max_word_num, analyzer=split_words_by_space)
    # descending sorting the dictionary by df
    X = vectorizer.fit_transform(doc_list)
    X[X > 0] = 1
    df = np.ravel(X.sum(axis=0))
    words = vectorizer.get_feature_names()
    word_df = list(zip(words, df))
    word_df.sort(key=lambda x: x[1], reverse=True)
    index2word = [w for (w, f) in word_df]
    index2word.extend(['<UNK>', '<GO>', '<EOS>', '<PAD>'])
    word2index = {w: i for i, w in enumerate(index2word)}

    return word2index, index2word


def format_train_data(u_i_r_p_m_c_tuple_list):
    user_set = set()
    item_set = set()
    comp_set = set()
    month_set = set()
    city_set = set()

    for x in u_i_r_p_m_c_tuple_list:
        user_set.add(x[0])
        item_set.add(x[1])
        comp_set.add(x[3])
        month_set.add(x[4])
        city_set.add(x[5])

    index2user = list(user_set)
    index2item = list(item_set)
    index2comp = list(comp_set)
    index2month = list(month_set)
    index2city = list(city_set)

    user2index = {x: i for i, x in enumerate(index2user)}
    item2index = {x: i for i, x in enumerate(index2item)}
    comp2index = {x: i for i, x in enumerate(index2comp)}
    month2index = {x: i for i, x in enumerate(index2month)}
    city2index = {x: i for i, x in enumerate(index2city)}

    new_u_i_r_p_m_c_tuple_list = []
    for x in u_i_r_p_m_c_tuple_list:
        new_u_i_r_p_m_c_tuple_list.append((user2index[x[0]], item2index[x[1]], x[2], comp2index[x[3]], month2index[x[4]], city2index[x[5]]))

    return new_u_i_r_p_m_c_tuple_list, index2user, index2item, index2comp, index2month, index2city


def get_now_time():
    """a string of current time"""
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def generate_word_pairs(word_id_list, window_size, ratio=1.0):
    word_list_len = len(word_id_list) - window_size
    inputs = []  # train data
    output = []  # label

    sliding_sign = window_size
    while sliding_sign < word_list_len:
        one_input = word_id_list[(sliding_sign - window_size): sliding_sign]  # left side
        one_input.extend(word_id_list[(sliding_sign + 1): (sliding_sign + window_size + 1)])  # right side
        inputs.append(one_input)
        output.append(word_id_list[sliding_sign])  # target label at the center of the window
        sliding_sign += 1

    length = len(output) * ratio
    if length > 1:
        index = list(range(len(output)))
        index_set = set()
        while len(index_set) < length:
            index_set.add(random.choice(index))

        new_input = []
        new_output = []
        for idx in index_set:
            new_input.append(inputs[idx])
            new_output.append(output[idx])

        return new_input, new_output

    return inputs, output
