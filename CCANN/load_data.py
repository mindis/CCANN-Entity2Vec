from utils import *
import numpy as np
import pickle


def load_ccann_data(data_dir, user2index, item2index, comp2index, month2index, city2index):
    if not os.path.exists(data_dir):
        sys.exit(get_now_time() + 'invalid directory')
    else:
        print(get_now_time() + 'loading split data')

        def read_from_file(path, max_rating, min_rating):
            tuple_list = []
            with open(path, 'r', errors='ignore') as f:
                for line in f.readlines():
                    content = line.strip().split('::')
                    u = user2index[content[0]]
                    i = item2index[content[1]]
                    p = comp2index[content[4]]
                    m = month2index[content[5]]
                    c = city2index[content[6]]
                    r = float(content[2])
                    if max_rating < r:
                        max_rating = r
                    if min_rating > r:
                        min_rating = r
                    tuple_list.append((u, i, r, p, m, c))
            return tuple_list, max_rating, min_rating

        max_r = -1
        min_r = 1e10
        train_tuple_list, max_r, min_r = read_from_file(data_dir + 'train', max_r, min_r)
        valid_tuple_list, max_r, min_r = read_from_file(data_dir + 'valid', max_r, min_r)
        test_tuple_list, max_r, min_r = read_from_file(data_dir + 'test', max_r, min_r)

        return train_tuple_list, valid_tuple_list, test_tuple_list, max_r, min_r


def load_embeddings(dir):
    user2embeddings = pickle.load(open(dir + 'user2embeddings.pickle', 'rb'))
    item2embeddings = pickle.load(open(dir + 'item2embeddings.pickle', 'rb'))
    comp2embeddings = pickle.load(open(dir + 'comp2embeddings.pickle', 'rb'))
    month2embeddings = pickle.load(open(dir + 'month2embeddings.pickle', 'rb'))
    city2embeddings = pickle.load(open(dir + 'city2embeddings.pickle', 'rb'))

    index2user = list(user2embeddings.keys())
    user2index = {x: i for i, x in enumerate(index2user)}
    user_embedding = np.asarray(list(user2embeddings.values()), dtype=np.float32)

    index2item = list(item2embeddings.keys())
    item2index = {x: i for i, x in enumerate(index2item)}
    item_embedding = np.asarray(list(item2embeddings.values()), dtype=np.float32)

    index2comp = list(comp2embeddings.keys())
    comp2index = {x: i for i, x in enumerate(index2comp)}
    comp_embedding = np.asarray(list(comp2embeddings.values()), dtype=np.float32)

    index2month = list(month2embeddings.keys())
    month2index = {x: i for i, x in enumerate(index2month)}
    month_embedding = np.asarray(list(month2embeddings.values()), dtype=np.float32)

    index2city = list(city2embeddings.keys())
    city2index = {x: i for i, x in enumerate(index2city)}
    city_embedding = np.asarray(list(city2embeddings.values()), dtype=np.float32)

    return user_embedding, index2user, user2index, item_embedding, index2item, item2index, comp_embedding, index2comp, comp2index, month_embedding, index2month, month2index, city_embedding, index2city, city2index


def load_ccann_data_id(data_dir):
    if not os.path.exists(data_dir):
        sys.exit(get_now_time() + 'invalid directory')
    else:
        print(get_now_time() + 'loading split data')

        # collect all users id and items id
        user_set = set()
        item_set = set()
        comp_set = set()
        month_set = set()
        city_set = set()

        def read_from_file(path):
            with open(path, 'r', errors='ignore') as f:
                for line in f.readlines():
                    content = line.strip().split('::')
                    user_set.add(content[0])
                    item_set.add(content[1])
                    comp_set.add(content[4])
                    month_set.add(content[5])
                    city_set.add(content[6])

        read_from_file(data_dir + 'train')
        read_from_file(data_dir + 'valid')
        read_from_file(data_dir + 'test')

        # convert id to array index
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

        def read_from_file2(path, max_rating, min_rating):
            tuple_list = []
            with open(path, 'r', errors='ignore') as f:
                for line in f.readlines():
                    content = line.strip().split('::')
                    u = user2index[content[0]]
                    i = item2index[content[1]]
                    p = comp2index[content[4]]
                    m = month2index[content[5]]
                    c = city2index[content[6]]
                    r = float(content[2])
                    if max_rating < r:
                        max_rating = r
                    if min_rating > r:
                        min_rating = r
                    tuple_list.append((u, i, r, p, m, c))
            return tuple_list, max_rating, min_rating

        max_r = -1
        min_r = 1e10
        train_tuple_list, max_r, min_r = read_from_file2(data_dir + 'train', max_r, min_r)
        valid_tuple_list, max_r, min_r = read_from_file2(data_dir + 'valid', max_r, min_r)
        test_tuple_list, max_r, min_r = read_from_file2(data_dir + 'test', max_r, min_r)

        return train_tuple_list, valid_tuple_list, test_tuple_list, max_r, min_r, index2user, index2item, index2comp, index2month, index2city
